# PokéLLMon

[//]: # (<div align="center">)

[//]: # (  <img src="./resource/LLM_attrition_strategy.gif" alt="PokemonBattle">)

[//]: # (</div>)

[//]: # ()

### Requirements:

```sh
python >= 3.8
openai >= 1.7.2
``` 

### Setting up a local battle engine

1. Install Node.js v10+.
2. Clone the Pokémon Showdown repository and set it up:

```sh
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
npm install
cp config/config-example.js config/config.js
node pokemon-showdown start --no-security
Enter "http://localhost:8000/" in your browsers.
``` 

### Configuring OpenAI API

Get OPENAI API from https://platform.openai.com/account/api-keys

```sh
export OPENAI_API_KEY=<your key>
```

### Showdown Calculator SeverSetup & Run
1. Navigate into the server directory and install the required Node.js dependencies
    - Before setup, please review the `dependencies` and `devDependencies` sections in the `calculator-server/package.json` file to ensure no unexpected libraries are present.
2. The server can be run using the pre-configured `start` script defined in the `calculator-server/package.json` file. This script uses `ts-node` to execute the TypeScript code directly.
```sh
cd calculator-server
npm install
npm start
```

### Local Battles
.env file would like to be
OPENAI_API_KEY="sk-proj-****"

```sh
C:/Users/gimhc/Github/PokeLLMon/.venv/Scripts/python.exe -m src.main2 --n_battles 1 --backend gpt-4o-mini-2024-07-18 --fast_model gpt-4o-mini-2024-07-18 --deep_model gpt-5.1-2025-11-13
``` 

### Suggestions
assume .venv is at root (Python 3.11.9)
```sh
pip install -r requirements.txt
$env:PYTHONPATH = ".."; $env:OPENAI_API_KEY = "OPENAI_API_KEY"; python main.py
```


