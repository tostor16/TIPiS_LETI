# Пояснение к домашней работе: идем в функцию bot.py/check_with_ai

    меняем переменную prompt
    меняем параметры (как минимум temperature), модель (см. на сайте c пометкой free)

openrouter.chat.completions.create(
            model="anthropic/claude-3.5-sonnet",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.3
        )

Есть и другие параметры, подробнее в документации, например

    top_p=1.0,                         # Nucleus sampling (0.0-1.0)
    frequency_penalty=0.0,              # Штраф за повторы (-2.0 до 2.0)
    presence_penalty=0.0,               # Штраф за упоминания (-2.0 до 2.0)

Запуск бота

    Проверьте python

# Откройте cmd (Win+R → cmd)
python --version
# или
python -V

# Если не работает, попробуйте:
py --version
py -V

# Показать все установленные версии:
py -0

    Клонируйте репозиторий с кодом.

    Получите токены и создайте .env Telegram бот:

    Найти @BotFather в Telegram
    Отправить /newbot
    Выбрать имя бота
    Скопировать токен

OpenRouter API: https://openrouter.ai

    Пройдите регистрацию

    Settings → Keys → Create Key Скопируйте ключ

Создайте .env:

TELEGRAM_BOT_TOKEN=твой_токен_бота
OPENROUTER_API_KEY=твой_ключ_openrouter

    Настрока виртуального окружения (опционально)

Для Windows (Откройте командную строку (Win+R → cmd))

# Перейдите в папку проекта
cd C:\путь\к\telegram_lab_bot

# Создайте виртуальное окружение
python -m venv venv

# Активируйте окружение
venv\Scripts\activate

# Должно появиться (venv) в начале строки:
(venv) C:\путь\к\telegram_lab_bot>

Для Linux все так же, кроме активации окружения:

source venv/bin/activate

    Установите библиотеки

# Обновите pip
python -m pip install --upgrade pip

# Установите все зависимости
pip install -r requirements.txt

    Запуск бота

python bot.py
