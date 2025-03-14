# Setting Up Your Workspace

## Source code
Download the release.zip file and extract it. This will include relevant source code from the game in `game`, along with documentation in the form of html files. To view the overall documentation for the game functions go to `docs/index.html`. Rules for the game and competition are also included in markdown files in `docs/rules`. To enable python to recognize the game imports, you'll need to set up your workspace to include game as part of your PATH. Below will show a video on how to do this in VS Code.

In your `workspace` folder, there will be a sample player which shows how to call functions.

## Prebuilt Clients
See if your architecture and OS combination has a prebuilt client:
* Windows 11, x64
* Mac Sierra 10.12+, ARM (M-chip) - NOTE: after opening, it might not be able to verify. See https://support.apple.com/guide/mac-help/open-a-mac-app-from-an-unknown-developer-mh40616/mac.
* Ubuntu Linux 22.04+, x64

If it doesn't, see the Unbuilt Clients section for how to build the client for your computer.

If it is, you can simply download the associated zip/installer for your computer. If the prebuilt clients don't work, see the following section for how to build the client for your computer.

## Unbuilt Clients
If your OS and architecture combination is not included in the prebuilt clients, don't worry! The process to build the client for your own computer isn't complicated. The following sections give instructions for how to build for Windows, Mac, and Linux. They are accompanied by videos of the process.

Note that as an alternative to any pip requirements in the below instructions, you may use anaconda instead and launch the application from your anaconda terminal.

If you have any issues with installing, developers can meet with you and help you build a version of the client for your architecture. 

## Windows

Install Python 3.10 from the bottom of the page here: https://www.python.org/downloads/release/python-31015/. **Make sure to check the box that says add Python to PATH.**

Install Node from the .msi here: https://nodejs.org/en/download

Download the unbuilt client from the ByteFight Downloads page and unzip it.

Open a cmd terminal as Administrator inside the directory that you unzipped (the directory should be `unbuilt-client`)


**Pip install**  
Run the following commands and respond yes to any prompts that pop up (this might take a little time to complete executing):
```
py -m pip install --upgrade pip
cd BotFightRenderer
pip install -r engine\requirements.txt
npm install
npm run electron:build
```

If all the previous steps ran without failing, in the folder `BotFightRenderer\dist\win-unpacked` should exist `ByteFight Client 2025.exe` and its associated libraries. Double click to start. If you prefer to install the client as an app, then there should be a `exe` installer in `BotFightRenderer\dist`.

**Anaconda Install**  
Alternatively, if you'd prefer to use Anaconda instead of pip, run the following in your terminal, again respond yes to any prompts:
```
cd BotFightRenderer
npm install
npm run electron:build
```

Next, in your Anaconda terminal, under the same directory (BotFightRenderer), run:
```
conda create --name bytefight python=3.10.5 --file engine\requirements.txt --channel conda-forge
conda activate bytefight
```

Now, cd to `BotFightRenderer\dist\win-unpacked` and, in your Anaconda prompt run `"ByteFight Client 2025.exe"` with quotes to launch the client. You will need to make sure your `bytefight` conda environment is active every time you start your client this way.

## Mac
Install Python 3.10 from the bottom of the page here: https://www.python.org/downloads/release/python-31015/. **Make sure to check the box that says add Python to PATH.**

Install Node from the .pkg here: https://nodejs.org/en/download

Download the unbuilt client from the ByteFight Downloads page and unzip it.

Open a terminal inside the directory that you unzipped (the directory should be `unbuilt-client`)

**Pip install**  
Run the following commands and respond yes to any prompts that pop up (this might take a little time to complete executing):
```
python3 -m pip install --upgrade pip
cd BotFightRenderer
pip3 install -r engine/requirements.txt
npm install
npm run electron:build
```

If all the previous steps ran without failing, in the folder `BotFightRenderer/dist/mac-unpacked` should exist a runnable `ByteFight Client 2025` that you can double-click and its associated libraries. If you prefer to install the client as an app, then there should be an installer in `BotFightRenderer/dist`.

**Anaconda Install**  
Alternatively, if you'd prefer to use Anaconda instead of pip, run the following in your terminal, again respond yes to any prompts:
```
cd BotFightRenderer
npm install
npm run electron:build
conda create --name bytefight python=3.10.5 --file engine/requirements.txt --channel conda-forge
conda activate bytefight
chmod +x "dist/mac-unpacked/ByteFight Client 2025"
```

Now, cd to`BotFightRenderer/dist/mac-unpacked` and run `"./ByteFight Client 2025"` with quotes to launch the client. You will need to make sure your `bytefight` conda environment is active every time you start your client this way.

## Linux  
Install Python 3.10 using 

```
sudo apt-get install python3.10
```
Install Node using commands from here: https://nodejs.org/en/download

Download the unbuilt client from the ByteFight Downloads page and unzip it.

Open a terminal inside the directory that you unzipped (the directory should be `unbuilt-client`)

**Pip install**  
Run the following commands and respond yes to any prompts that pop up (this might take a little time to complete executing):
```
sudo apt-get install rpm
python3 -m pip install --upgrade pip
cd BotFightRenderer
pip3 install -r engine/requirements.txt
npm install
npm run electron:build
```

If all the previous steps ran without failing, in the folder `BotFightRenderer/dist/inux-unpacked` should exist the runnable `bytefight-client-2025` that you can double click and its associated libraries. If you prefer to install the client as an app, then there should be an `AppImage` and `snap` installer in `BotFightRenderer/dist`.

**Anaconda Install**  
Alternatively, if you'd prefer to use Anaconda instead of pip, run the following in your terminal, respond yes to any prompts:
```
sudo apt-get install rpm
cd BotFightRenderer
npm install
npm run electron:build
conda create --name bytefight python=3.10.5 --file engine/requirements.txt --channel conda-forge
conda activate bytefight
chmod +x dist/linux-unpacked/bytefight-client-2025
```

Now, cd to`BotFightRenderer/dist/linux-unpacked` and run `./bytefight-client-2025`. You will need to make sure your `bytefight` conda environment is active every time you start your client this way.