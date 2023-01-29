from aiogram.dispatcher.filters.state import State, StatesGroup


class FSMBot(StatesGroup):
    expect_content = State()
    expect_style = State()
