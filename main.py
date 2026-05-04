import os
import asyncio
import pandas as pd
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types, F
from aiogram.types import FSInputFile
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
SAMBANOVA_API_KEY = os.getenv("SAMBANOVA_API_KEY")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

llm = ChatOpenAI(
    model='Meta-Llama-3.3-70B-Instruct', 
    openai_api_key=SAMBANOVA_API_KEY,
    openai_api_base="https://api.sambanova.ai/v1",
    temperature=0.1
)

TEMP_DIR = "temp_files"
os.makedirs(TEMP_DIR, exist_ok=True)

@dp.message(F.document)
async def handle_analysis(message: types.Message):
    file_name = message.document.file_name
    if not (file_name.endswith('.csv') or file_name.endswith('.xlsx')):
        await message.answer("❌ Пожалуйста, отправь файл .csv или .xlsx")
        return

    wait_msg = await message.answer("🚀 Начинаю агентный анализ данных (SambaNova Llama 3.3)...")

    file_id = message.document.file_id
    file_info = await bot.get_file(file_id)
    local_path = os.path.join(TEMP_DIR, f"{file_id}_{file_name}")
    await bot.download_file(file_info.file_path, local_path)

    try:
        df = pd.read_csv(local_path) if file_name.endswith('.csv') else pd.read_excel(local_path)
        agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            allow_dangerous_code=True,
            agent_type="openai-tools", 
        )

        plot_path = os.path.join(TEMP_DIR, f"plot_{file_id}.png")
        
        # Инструкция для агента
        query = (
            "ТЫ — ПРОФЕССИОНАЛЬНЫЙ АНАЛИТИК ДАННЫХ. ВАЖНО: ОТВЕЧАЙ СТРОГО НА РУССКОМ ЯЗЫКЕ. "
            "1. Проведи глубокий анализ датасета. "
            "2. Напиши 3 ключевых инсайда на русском. "
            f"3. Построй график matplotlib и сохрани его по пути '{plot_path}'. "
            "В текстовом ответе не пиши путь к файлу и технические детали установки библиотек, "
            "просто дай аналитический отчет."
        )

        response = await asyncio.to_thread(agent.invoke, {"input": query})
        result_text = response["output"]

        await message.answer(f"📊 **Аналитический отчет:**\n\n{result_text}", parse_mode="Markdown")

        # Отправка графика, если агент его создал
        if os.path.exists(plot_path):
            await bot.send_photo(message.chat.id, FSInputFile(plot_path), caption="Визуализация данных")
            os.remove(plot_path)

    except Exception as e:
        await message.answer(f"❌ Ошибка при анализе: {str(e)}")
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)
        await wait_msg.delete()

@dp.message()
async def cmd_start(message: types.Message):
    await message.answer("👋 Привет! Я твой ИИ-аналитик. Пришли мне CSV или Excel файл, и я изучу его с помощью Python-кода.")

async def main():
    print("Бот запущен на SambaNova (Llama 3.3)!")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())