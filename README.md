# california_housing

# software and tool required

[Github Account](https://github.com)
[vs code ide]
[gitCLI]

(Getting-started-with commondline)


create a new environment  for this project

conda create -p venv python--3.7 -y

## Run with Streamlit
To run the Streamlit app locally (recommended), create and activate your virtual environment, install dependencies, and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

For deployment to Streamlit Cloud, push the repository to GitHub and connect the repo in Streamlit Cloud. Ensure `requirements.txt` contains all dependencies.