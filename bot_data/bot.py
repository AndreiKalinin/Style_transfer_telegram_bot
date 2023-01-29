from aiogram import Bot, Dispatcher, executor, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext

import os
import dotenv
import io
from bot_data.states import FSMBot
from model_data.model import StyleTransfer


dotenv.load_dotenv()
token = os.getenv('BOT_TOKEN')

storage = MemoryStorage()
bot = Bot(token)
dp = Dispatcher(bot, storage=storage)
content_image_buffer = io.BytesIO()


@dp.message_handler(state='*', commands=['start'])
async def process_start_command(message: types.Message):
    """The bot responds to start command and waits for an original image"""
    await message.answer('Привет!\nЯ умею переносить стиль с одного изображения на другое.\n'
                         'Чтобы начать, пришли мне исходное изображение.\n'
                         'Для получения справки отправь команду /help')
    await FSMBot.expect_content.set()


@dp.message_handler(state='*', commands=['help'])
async def process_help_command(message: types.Message):
    """Respond to help command"""
    await message.answer('Сначала необходимо отправить исходное изображение, а затем изображение, '
                         'стиль которого нужно перенести')


@dp.message_handler(state='*', commands=['cancel'])
async def process_cancel_command(message: types.Message):
    """Respond to cancel command"""
    await message.answer('Чтобы начать, пришли мне исходное изображение.\n'
                         'Для получения справки отправь команду /help')
    await FSMBot.expect_content.set()


@dp.message_handler(state=FSMBot.expect_content, content_types=['photo'])
async def process_content_upload(message: types.Message):
    """The bot gets an original image and asks for a style image"""
    global content_image_buffer
    await message.photo[-1].download(destination_file=content_image_buffer)
    await message.answer('Теперь пришли изображение, стиль которого необходимо применить')
    await FSMBot.expect_style.set()


@dp.message_handler(state=FSMBot.expect_content, content_types=['any'])
async def content_warning_wrong_format(message: types.Message):
    """The bot asks for an image if it gets the wrong data input while being in the expect_content state"""
    await message.answer('Пожалуйста, пришли изображение. Для помощи отправь команду /help, для отмены - /cancel')


@dp.message_handler(state=FSMBot.expect_style, content_types=['photo'])
async def process_style_upload(message: types.Message, state: FSMContext):
    """The bot receives a style photo, launches style transfer and sends an output image"""
    global content_image_buffer
    style_image_buffer = io.BytesIO()
    await message.photo[-1].download(destination_file=style_image_buffer)
    await message.answer('Генерирую изображение, это займет некоторое время...')
    style_transfer = StyleTransfer(content_image_buffer, style_image_buffer)
    output_name = f'output_{message.from_user.id}.png'
    style_transfer.save_image(output_name)
    await message.answer_photo(types.InputFile(output_name))
    await state.finish()
    await message.answer('Готово!\nЧтобы начать с начала, отправь /start')
    content_image_buffer = io.BytesIO()
    os.remove(output_name)


@dp.message_handler(state=FSMBot.expect_style, content_types=['any'])
async def style_warning_wrong_format(message: types.Message):
    """The bot asks for an image if it gets the wrong data input while being in the expect_style state"""
    await message.answer('Пожалуйста, пришли изображение. Для помощи отправь команду /help, для отмены - /cancel')


def launch_bot():
    """Launch bot polling"""
    executor.start_polling(dp, skip_updates=True)
