
<h1 align='center'>
   <img src = 'https://github.com/ClementPerroud/Gym-Trading-Env/raw/main/docs/source/images/logo_light-bg.png' width='500'>
</h1>

<section class="shields" align="center">
   <a href="https://www.python.org/">
      <img src="https://img.shields.io/badge/python-v3-brightgreen.svg"
         alt="python">
   </a>
   <a href="https://pypi.org/project/gym-trading-env/">
      <img src="https://img.shields.io/badge/pypi-v1.1.3-brightgreen.svg"
         alt="PyPI">
   </a>
   <a href="https://github.com/ClementPerroud/Gym-Trading-Env/blob/main/LICENSE.txt">
   <img src="https://img.shields.io/badge/license-MIT%202.0%20Clause-green"
         alt="Apache 2.0 with Commons Clause">
   </a>
   <a href='https://gym-trading-env.readthedocs.io/en/latest/?badge=latest'>
         <img src='https://readthedocs.org/projects/gym-trading-env/badge/?version=latest' alt='Documentation Status' />
   </a>
   <a href="https://github.com/ClementPerroud/Gym-Trading-Env">
      <img src="https://img.shields.io/github/stars/ClementPerroud/gym-trading-env?style=social" alt="Github stars">
   </a>
</section>
  
# Gym Trading Env

Entorno basado en Gymnasium para simular mercados y entrenar agentes de Aprendizaje por Refuerzo (RL) en trading. Está diseñado para ser rápido, simple de usar y altamente personalizable.

| [Documentación](https://gym-trading-env.readthedocs.io/en/latest/index.html) |

## Características clave

- Descarga rápida de datos OHLCV desde múltiples exchanges vía `ccxt`.
- Entornos simples y veloces que soportan operaciones avanzadas (short, interés por préstamo, comisiones) con posiciones continuas o discretas.
- Renderizador web de alto rendimiento para visualizar miles de velas, posiciones y métricas personalizadas.
- Soporte para múltiples datasets y ejecución vectorizada con la API de Gymnasium.

Imagen de ejemplo de renderizado:

![Render animated image](https://raw.githubusercontent.com/ClementPerroud/Gym-Trading-Env/main/docs/source/images/render.gif)

## IDs de entornos registrados

Los entornos se registran en `src/gym_trading_env/__init__.py` y pueden crearse con `gym.make(...)` o `gym.make_vec(...)`:

- `TradingEnv`: entorno para un único dataset.
- `MultiDatasetTradingEnv`: entorno que itera sobre múltiples datasets preprocesados.

## Instalación

Requiere Python 3.9+ (Linux, macOS o Windows).

- Opción 1 (PyPI):

```bash
pip install gym-trading-env
```

- Opción 2 (clonando el repo):

```bash
git clone https://github.com/ClementPerroud/Gym-Trading-Env
cd Gym-Trading-Env
```

Si usas Poetry para gestionar dependencias del proyecto:

```bash
poetry install
```

## Formato de datos de entrada

Los DataFrames deben tener índice `DatetimeIndex` y, como mínimo, las columnas: `open`, `high`, `low`, `close` y `volume` (o `Volume USD` en algunos ejemplos). Las características (features) que formarán parte de la observación deben incluir la palabra `feature` en su nombre, por ejemplo: `feature_close`, `feature_volume`, etc.

## Ejemplo rápido (single dataset)

```python
import gymnasium as gym
import numpy as np
import pandas as pd

# Cargar datos
df = pd.read_csv("examples/data/BTC_USD-Hourly.csv", parse_dates=["date"], index_col="date").sort_index()
df = df.dropna().drop_duplicates()

# Crear features (nota: deben contener la palabra 'feature')
df["feature_close"] = df["close"].pct_change()
df["feature_open"] = df["open"] / df["close"]
df["feature_high"] = df["high"] / df["close"]
df["feature_low"] = df["low"] / df["close"]
df["feature_volume"] = df["Volume USD"] / df["Volume USD"].rolling(7 * 24).max()
df.dropna(inplace=True)

def reward_function(history):
    # Recompensa por log-retorno del valor del portafolio
    return np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2])

env = gym.make(
    "TradingEnv",
    name="BTCUSD",
    df=df,
    windows=5,
    positions=[-1, -0.5, 0, 0.5, 1, 1.5, 2],
    initial_position="random",
    trading_fees=0.01 / 100,
    borrow_interest_rate=0.0003 / 100,
    reward_function=reward_function,
    portfolio_initial_value=1000,
    max_episode_duration=500,
)

obs, info = env.reset()
done, truncated = False, False
while not done and not truncated:
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)

# Guardar logs para el renderizador
env.save_for_render()
```

Consulta `examples/example_environnement.py` para un script completo.

## Ejecución vectorizada (single dataset)

```python
import gymnasium as gym
import numpy as np
import pandas as pd

# Preparar df y reward_function como en el ejemplo anterior...

env = gym.make_vec(
    id="TradingEnv",
    num_envs=3,
    name="BTCUSD",
    df=df,
    windows=5,
    positions=[-1, -0.5, 0, 0.5, 1, 1.5, 2],
    initial_position=0,
    trading_fees=0.01 / 100,
    borrow_interest_rate=0.0003 / 100,
    reward_function=reward_function,
    portfolio_initial_value=1000,
)

obs, info = env.reset()
while True:
    actions = [1, 2, 3]
    obs, reward, done, truncated, info = env.step(actions)
```

Consulta `examples/example_vectorized_environment.py`.

## Múltiples datasets (con preprocesamiento)

```python
import gymnasium as gym
import numpy as np
import pandas as pd

def preprocess(df: pd.DataFrame):
    df["feature_close"] = df["close"].pct_change()
    df["feature_open"] = df["open"] / df["close"]
    df["feature_high"] = df["high"] / df["close"]
    df["feature_low"] = df["low"] / df["close"]
    df["feature_volume"] = df["volume"] / df["volume"].rolling(7 * 24).max()
    df.dropna(inplace=True)
    return df

def reward_function(history):
    return np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2])

env = gym.make(
    "MultiDatasetTradingEnv",
    dataset_dir="examples/data/*.pkl",
    preprocess=preprocess,
    windows=5,
    positions=[-1, -0.5, 0, 0.5, 1, 1.5, 2],
    initial_position=0,
    trading_fees=0.01 / 100,
    borrow_interest_rate=0.0003 / 100,
    reward_function=reward_function,
    portfolio_initial_value=1000,
    max_episode_duration=500,
    episodes_between_dataset_switch=10,
)
```

Versiones vectorizadas: `examples/example_vectorized_multi_environment.py`.

## Descarga de datos (ccxt)

Usa `gym_trading_env.downloader.download` para descargar OHLCV y guardar `.pkl` listos para usar:

```python
import datetime as dt
from gym_trading_env.downloader import EXCHANGE_LIMIT_RATES, download

# Opcional: ajustar límites por exchange
EXCHANGE_LIMIT_RATES["bybit"] = {"limit": 200, "pause_every": 120, "pause": 2}

download(
    exchange_names=["binance"],
    symbols=["BTC/USDT"],
    timeframe="1h",
    dir="examples/data",
    since=dt.datetime(2023, 1, 1),
)
```

Consulta `examples/example_download.py`.

## Renderizador web

```python
from gym_trading_env.renderer import Renderer

renderer = Renderer(render_logs_dir="render_logs")

# Líneas y métricas personalizadas (opcional)
# renderer.add_line("sma10", lambda df: df["close"].rolling(10).mean(), {"width": 1, "color": "purple"})
# renderer.add_metric("Annual Market Return", lambda df: "...")

renderer.run()
```

Consulta `examples/example_render.py`.

## Cómo ejecutar los ejemplos

Desde la raíz del repo:

```bash
# Con Poetry (recomendado en este repo)
poetry run python examples/example_environnement.py
poetry run python examples/example_vectorized_environment.py
poetry run python examples/example_multi_environnement.py
poetry run python examples/example_vectorized_multi_environment.py
poetry run python examples/example_render.py
```

También puedes ejecutar con `python` directamente si ya tienes las dependencias instaladas en tu entorno actual.

## Estructura relevante del proyecto

- `src/gym_trading_env/`: implementación principal
  - `environments.py`: clases `TradingEnv` y `MultiDatasetTradingEnv`
  - `downloader.py`: utilidades asíncronas para descargar OHLCV (`ccxt`)
  - `renderer.py`: servidor Flask para visualizar sesiones
  - `utils/`: historial, portafolio, y utilidades de gráficos
- `examples/`: scripts listos para ejecutar que cubren descarga, ejecución, vectorización y renderizado

## Licencia

Este proyecto se distribuye bajo la licencia MIT. Consulta `LICENSE.txt` para más detalles.

[Documentation available here](https://gym-trading-env.readthedocs.io/en/latest/index.html)
