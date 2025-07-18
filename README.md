# Vogue-Vision
A PyTorch-powered multi-output CNN that classifies fashion images by type, color, season, and gender. Deployed with FastAPI and Streamlit.


## What is this Project about?

This project presents a deep learning solution for automated fashion product analysis. It utilizes a multi-output convolutional neural network (CNN) built with PyTorch to predict four key attributes of a fashion item from its image: the **product type**, the **base color**, the appropriate **season** for use, and the target **gender**. The project includes the entire pipeline from data exploration and model training in a Jupyter Notebook to deployment as both a REST API with FastAPI and an interactive web application with Streamlit.

## Tools and Libraries Used

* **Programming Language:** Python 3.10+
* **Deep Learning:** PyTorch
* **Data Manipulation & Analysis:** Pandas, NumPy, Scikit-learn
* **Data Visualization:** Matplotlib, Seaborn
* **Web API:** FastAPI, Uvicorn
* **Web Application GUI:** Streamlit
* **Core Libraries:** Pillow (for image processing), TQDM (for progress bars)

## How to Go About the Project

Follow these steps to set up and run the project on your local machine,

### Step 1: Clone the Repository

```bash
git clone https://github.com/sumukhacharya03/Vogue-Vision.git
cd Vogue-Vision
```

### Step 2: Set Up a Virtual Environment

I would recommended to create a virtual environment to keep the project dependencies isolated

```bash
# Create a virtual environment named 'venv'
python -m venv venv

# Activate the virtual environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

Install all the required dependencies using the `requirements.txt` file

```bash
pip install -r requirements.txt
```

### Step 4: Running the Applications

You can explore the project in three ways:

#### A. Explore the Training Process (Jupyter Notebook)

The `vogue-vision.ipynb` file contains the complete end-to-end process, including data loading, exploratory data analysis (EDA), data preprocessing, model building, training and testing. To run this, you will need to install Jupyter Notebook:

```bash
# Install Jupyter if you haven't already
pip install jupyter

# Launch Jupyter Notebook
jupyter notebook

# Then run the ipynb file named "vogue-vision.ipynb" in the browser window
```
*Note: The notebook is set up for the Kaggle environment. You may need to adjust file paths to run it locally.*

#### B. Run the FastAPI Server (API)

The API allows you to get predictions programmatically,

```bash
# Start the Uvicorn server with auto-reload
uvicorn api:app --reload
```

Once running, open your web browser and navigate to **`http://127.0.0.1:8000/docs`** to access the interactive API documentation and test the `/predict/` endpoint.

#### C. Run the Streamlit Web App (GUI)

The Streamlit app provides a user-friendly graphical interface to test the model,

* Open a **new terminal** in the project directory.
* Activate the virtual environment again (`.\venv\Scripts\activate`).
* Run the following command:

```bash
streamlit run app.py
```

This will automatically open a new tab in your browser with the web application. You can then upload an image and get predictions instantly.
