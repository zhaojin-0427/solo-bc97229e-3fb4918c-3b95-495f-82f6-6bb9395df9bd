"""并发与一致性：抽取期间持续增删改，切片仍来自同一快照。"""
from __future__ import annotations

import sqlite3
import threading

from app.demo import DemoWriter
from app.extractor import run_extraction


def _write_source(source_db, statements):
    """以开启外键约束的单事务执行一组合法更新，返回是否已提交。"""
    conn = sqlite3.connect(str(source_db), timeout=30)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("BEGIN")
        for sql, params in statements:
            conn.execute(sql, params)
        conn.commit()
    finally:
        conn.close()
    return True


def test_snapshot_isolation_under_concurrent_writes(
    source_db, schema, out_path
):
    """快照固定后写入器疯狂增删改，切片内容与快照时刻完全一致。"""
    writer = DemoWriter(str(source_db), interval=0.05)
    fired = threading.Event()

    def after_snapshot():
        # 快照读标记固定后才启动写入，保证写入都发生在遍历/拷贝期间
        writer.start()
        for _ in range(20):
            writer.tick_once()
        fired.set()

    result = run_extraction(
        str(source_db),
        schema,
        1,
        out_path,
        batch_size=1,  # 极小批次，尽量拉长遍历窗口
        on_snapshot=after_snapshot,
    )
    fired.wait(5)
    writer.stop()

    # 源库已被写入器改动（出现 900x 演示单据或状态变化）
    live = sqlite3.connect(str(source_db))
    demo_rows = live.execute(
        "SELECT count(*) FROM return_orders WHERE id >= 9000"
    ).fetchone()[0]
    live.close()
    assert demo_rows >= 1

    # 切片只含快照时刻的退货单 1，看不到任何演示单据
    dst = sqlite3.connect(str(out_path))
    try:
        assert dst.execute(
            "PRAGMA foreign_key_check"
        ).fetchall() == []
        ids = {
            r[0]
            for r in dst.execute("SELECT id FROM return_orders")
        }
        assert ids == {1}
        assert result.row_counts["return_orders"] == 1
        # 快照时刻客户只有 1..4，切片不允许出现后来的任何变化
        assert dst.execute(
            "SELECT count(*) FROM return_orders WHERE id >= 9000"
        ).fetchone()[0] == 0
    finally:
        dst.close()


def test_source_wal_mode_supported(source_db):
    conn = sqlite3.connect(str(source_db))
    try:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"
    finally:
        conn.close()


def test_copy_phase_reads_single_snapshot(source_db, schema, out_path):
    """客户表拷贝完成后、商品表拷贝前，单事务同时改客户名与商品价格。

    切片必须仍含*旧*客户名与*旧*价格（同一快照），不得拼出
    “旧客户名 + 新价格”；写入事务必须在抽取结束前正常提交。
    交错时序由 on_event 钩子精确控制，无 sleep、无随机性。
    """
    customers_copied = threading.Event()
    writer_committed = threading.Event()

    def on_event(event, info):
        # 拷贝顺序 customers -> products：在两者之间插入并发写入
        if event == "after_copy_table" and info["table"] == "customers":
            customers_copied.set()
            # 若快照连接阻塞写入，这里将超时 -> 测试失败而非死等
            assert writer_committed.wait(timeout=10), (
                "源库写入未能在抽取期间提交（快照连接阻塞了写入）"
            )

    def writer():
        try:
            assert customers_copied.wait(timeout=10)
            _write_source(
                source_db,
                [
                    ("UPDATE customers SET name = ? WHERE id = 1",
                     ("张三(抽取期间改名)",)),
                    ("UPDATE products SET price_cents = 1 WHERE id = 1",
                     ()),
                ],
            )
        finally:
            writer_committed.set()

    t = threading.Thread(target=writer, name="test-writer")
    t.start()
    result = run_extraction(
        str(source_db), schema, 1, out_path, on_event=on_event
    )
    t.join(timeout=10)

    # 写入确实在抽取结束前已提交
    assert writer_committed.is_set()
    live = sqlite3.connect(str(source_db))
    try:
        assert live.execute(
            "SELECT name FROM customers WHERE id = 1"
        ).fetchone()[0] == "张三(抽取期间改名)"
        assert live.execute(
            "SELECT price_cents FROM products WHERE id = 1"
        ).fetchone()[0] == 1
    finally:
        live.close()

    # 切片完全来自快照时刻：旧名 + 旧价，外键检查通过
    dst = sqlite3.connect(str(out_path))
    try:
        assert dst.execute("PRAGMA foreign_key_check").fetchall() == []
        assert dst.execute(
            "SELECT name FROM customers WHERE id = 1"
        ).fetchone()[0] == "张三"
        assert dst.execute(
            "SELECT price_cents FROM products WHERE id = 1"
        ).fetchone()[0] == 29900
    finally:
        dst.close()
    assert result.row_counts["customers"] == 2
    assert result.row_counts["products"] == 1


def test_post_traversal_update_no_false_fk_failure(
    source_db, schema, out_path
):
    """闭包遍历完成后，明细改指另一商品并删除旧商品（合法单事务）。

    抽取必须成功：切片保留快照时刻的旧引用与旧商品行，外键检查通过，
    不得误报外键失败；写入事务必须在抽取结束前正常提交。
    """
    traversal_done = threading.Event()
    writer_committed = threading.Event()

    def on_event(event, info):
        if event == "traverse_done":
            traversal_done.set()
            assert writer_committed.wait(timeout=10), (
                "源库写入未能在抽取期间提交（快照连接阻塞了写入）"
            )

    def writer():
        try:
            assert traversal_done.wait(timeout=10)
            _write_source(
                source_db,
                [
                    # 退货明细 1 与订单行 1 改指商品 2，再删除商品 1
                    ("UPDATE return_lines SET product_id = 2 WHERE id = 1",
                     ()),
                    ("UPDATE order_lines SET product_id = 2 WHERE id = 1",
                     ()),
                    ("DELETE FROM products WHERE id = 1", ()),
                ],
            )
        finally:
            writer_committed.set()

    t = threading.Thread(target=writer, name="test-writer")
    t.start()
    # 修复前：此处抛出 ExtractionError（外键检查误报）
    result = run_extraction(
        str(source_db), schema, 1, out_path, on_event=on_event
    )
    t.join(timeout=10)

    assert writer_committed.is_set()
    live = sqlite3.connect(str(source_db))
    try:
        # 源库已是新状态：商品 1 被删，明细指向商品 2
        assert live.execute(
            "SELECT count(*) FROM products WHERE id = 1"
        ).fetchone()[0] == 0
        assert live.execute(
            "SELECT product_id FROM return_lines WHERE id = 1"
        ).fetchone()[0] == 2
    finally:
        live.close()

    # 切片仍是快照时刻：明细指向商品 1，商品 1 在切片中存在
    dst = sqlite3.connect(str(out_path))
    try:
        assert dst.execute("PRAGMA foreign_key_check").fetchall() == []
        assert dst.execute(
            "SELECT product_id FROM return_lines WHERE id = 1"
        ).fetchone()[0] == 1
        assert dst.execute(
            "SELECT price_cents FROM products WHERE id = 1"
        ).fetchone()[0] == 29900
    finally:
        dst.close()
    assert result.row_counts["products"] == 1


def test_repeated_extractions_identical(source_db, schema, tmp_path):
    """同一源库、同一退货单的两次切片逐字节结构一致、行数一致。"""
    out1 = tmp_path / "a.db"
    out2 = tmp_path / "b.db"
    r1 = run_extraction(str(source_db), schema, 1, out1)
    r2 = run_extraction(str(source_db), schema, 1, out2)
    assert r1.row_counts == r2.row_counts

    def dump(path):
        c = sqlite3.connect(str(path))
        lines = []
        for (sql,) in c.execute(
            "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL "
            "ORDER BY tbl_name, type"
        ):
            lines.append(sql)
        for table in (
            "return_orders",
            "return_lines",
            "orders",
            "order_lines",
            "customers",
            "products",
        ):
            rows = c.execute(
                f'SELECT * FROM "{table}" ORDER BY id'
            ).fetchall()
            lines.append((table, rows))
        c.close()
        return lines

    assert dump(out1) == dump(out2)
