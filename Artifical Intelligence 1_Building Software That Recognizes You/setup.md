# Workshop setup

To follow along with the exercises which make up this workshop, you will need a few utilities installed on your system, including Python and an IDE.

## Code Editor (Visual Studio Code)

To work with the code you will need a code editor. You are free to use any editor you like. Your instructor will use Visual Studio Code, an open source (OSS) editor from Microsoft.

### Installation

The [Visual Studio Code website](https://code.visualstudio.com/) will allow you to download and install Visual Studio Code.

## Python

The workshop presents a scenario where you will build a website using [Flask](https://palletsprojects.com/p/flask/), which is a lightweight [Python](https://python.org) framework.

### Installation

To install Python, navigate to [Python.org](https://python.org) and follow the instructions.

> **NOTE** If installing on Windows, ensure you select the option to add Python to your PATH system variable.

## Azure Command Line Interface (Azure CLI)

To ease creation of resources on Azure, this workshop uses the Azure CLI. By using the Azure CLI you are able to manage all Azure resources.

### Installation

To install the Azure CLI, navigate to [Azure CLI installation](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest) and follow the instructions. Beccause the Azure CLI is based on Python it will run on all operating systems.

Once installed, login to the Azure CLI by using `az login`. This operation will open a browser for authentication.

## Git

To download the starter and solution files, you will [clone](https://help.github.com/en/articles/cloning-a-repository) a repository from [GitHub](https://github.com) using git. Git is a distributed source code management system.

### Installation

To install git, navigate to [Git downloads](https://git-scm.com/downloads) and follow the instructions.

## Starter code

The sample code is provided as part of the [Reactors](https://github.com/microsoft/reactors) repository on [GitHub](https://github.com). Let's clone the repository and get the environment setup for the code.

### Clone the repository

1. Open a command or terminal window
2. Navigate to the folder you want to put the code into
3. Clone the repository

``` git
git clone https://github.com/microsoft/reactors
```

4. Navigate to the AI directory

``` console
# Windows
cd reactors\Artifical Intelligence 1_Building Software That Recognizes You\starter-site

# Linux or macOS
cd ./reactors/Artifical Intelligence 1_Building Software That Recognizes You/starter-site
```

5. Create a virtual environment and install packages
Let's create a virtual environment for the packages we'll be using. Virtual environments allow us to separate packages from other environments. Return to the command line and issuing the following command:

``` console
# Windows
python -m venv env
.\env\Scripts\activate

# macOS or Linux
python3 -m venv env
. ./env/bin/activate
```

Note: If you're using macOS or Linux the leading . for the . ./env/bin/activate is required as it indicates to Python where your source code resides.

6. Install the necessary Python packages
Install the packages listed in requirements.txt by using pip

pip install -r requirements.tx

7. Confirm the site starts

``` console
flask run
```

Navigate to **http://localhost:5000**

## You're all set!!
