import asyncio

from skyrl.train.fully_async_trainer import _AsyncDataloader, _resume_starts_new_epoch


class DummyStatefulDataLoader:
    def __init__(self, items):
        self._items = items
        self._index = 0

    def __iter__(self):
        while self._index < len(self._items):
            item = self._items[self._index]
            self._index += 1
            yield [item]

    def __len__(self):
        return len(self._items)

    def state_dict(self):
        return {"index": self._index}

    def load_state_dict(self, state):
        self._index = state["index"]


def test_resume_starts_new_epoch_only_at_epoch_boundary():
    assert _resume_starts_new_epoch(global_step=16, num_steps_per_epoch=4, consumed_uid_count=256) is True
    assert _resume_starts_new_epoch(global_step=15, num_steps_per_epoch=4, consumed_uid_count=192) is False
    assert _resume_starts_new_epoch(global_step=0, num_steps_per_epoch=4, consumed_uid_count=0) is False


def test_async_dataloader_reset_reopens_epoch_after_full_consumption():
    async def exercise():
        items = [{"uid": f"uid-{idx}"} for idx in range(4)]
        dataloader = DummyStatefulDataLoader(items)
        async_dataloader = _AsyncDataloader(dataloader, mini_batch_size=2)

        async_dataloader.load_state_from_checkpoint({item["uid"] for item in items})
        assert await async_dataloader.get_next_non_consumed_data() is None

        await async_dataloader.reset_at_epoch_end()
        next_item = await async_dataloader.get_next_non_consumed_data()
        assert next_item[0]["uid"] == "uid-0"

    asyncio.run(exercise())
