# Placement Predictor

A simple web app that estimates how likely a student is to get placed, using only two numbers: **IQ** and **CGPA**.

**Live app:** https://YOUR-APP-NAME.streamlit.app

> This is a learning project. It uses a small dataset of 100 students, so the result is an estimate, not a promise.

---

## What does this project do?

You enter a student's IQ and CGPA. The app gives you:

- A result: **Placed** or **Not placed**
- A probability, for example "72% chance"
- A confidence level: Low, Medium or High
- A simple explanation of why the model gave that answer

The app is made for two kinds of people:

1. **Students** who want to check their own chances.
2. **Teachers, placement cell staff and recruiters** who want to check many students at once.

---

## The four modes

### 1. Student mode
- Move the sliders (or type numbers) for IQ and CGPA, then press **Predict**.
- Try the quick example profiles: Strong profile, Borderline, Needs work.
- See which feature pushes the result up or down.
- See where you sit among all the students in the dataset.
- See a "what if" chart: how the chance changes if your CGPA goes up by 0.5.
- Pick a target (for example 70%) and see the CGPA you would need at your IQ.
- Download a one-page summary of your result.

### 2. Compare mode
- Put two students side by side.
- See who has the better chance, and whether IQ or CGPA explains the difference.

### 3. Batch mode
- Upload a CSV file with `IQ` and `CGPA` columns and get a prediction for every row.
- No file? Switch on the built-in sample data to try it.
- Change the decision threshold to be stricter or more relaxed.
- See a chart of all the probabilities and a Top 10 candidates table.
- Download the results as a CSV file.
- The app checks your file and shows friendly messages if columns are missing or values are wrong.

### 4. Insights mode
- A colour map showing the model's estimate across all IQ and CGPA values.
- Model accuracy on a test set, plus a steadier 5-fold cross-validation score.
- A confusion matrix (how many right and wrong guesses).
- Which feature matters more, explained in plain words.
- Placement rate for each CGPA band.

There is also a light/dark theme switch and a session history in the sidebar.

---

## How does it work?

1. **Data:** `ml_student_placement_100_rows.csv` has 100 students with three columns: IQ, CGPA and Placement (placed or not).
2. **Training:** The data is split 80% for training and 20% for testing.
3. **Scaling:** IQ and CGPA have very different sizes, so a `StandardScaler` puts them on the same scale before training.
4. **Model:** A **Logistic Regression** model learns from the scaled data. It gives a probability between 0% and 100%.
5. **Prediction:** The app scales every new input the same way, then asks the model for a probability. At 50% or higher, the result is "Placed".

The model and the scaler are saved as `model.pkl` and `scaler.pkl`, and the app loads them when it starts.

---

## Built with

- Python
- Streamlit (the web app)
- scikit-learn (the model and scaler)
- pandas and NumPy (data)
- Plotly (charts)

---

## Project files

```
placement-predictor/
├── app.py                  The Streamlit app
├── train.py                Trains the model and saves model.pkl and scaler.pkl
├── model.pkl               The trained model
├── scaler.pkl              The fitted scaler
├── ml_student_placement_100_rows.csv   The dataset
├── requirements.txt        Libraries the app needs
└── .streamlit/config.toml  App colours and settings
```

---

## Run it on your computer

1. Download or clone this project.
2. Open a terminal in the project folder and install the libraries:

```
pip install -r requirements.txt
```

3. Start the app:

```
streamlit run app.py
```

4. Open http://localhost:8501 in your browser.

### Train the model again (optional)

```
python train.py
```

This creates fresh `model.pkl` and `scaler.pkl` files and prints the test accuracy. Always keep the two files together, because the model only works with the scaler it was trained with.

---

## Batch file format

Your CSV needs at least these two columns (any other columns, like a name, are kept):

```
Student,IQ,CGPA
Student 1,98,6.5
Student 2,110,8.2
```

You can download a ready-made template inside the app.

---

## Important limits

- The model learned from only **100 students**, so the results can change a lot with new data.
- It only looks at **IQ and CGPA**. Real placements also depend on skills, projects, communication, interviews and the job market.
- It is an **estimate, not a guarantee**. Please do not use it for real admission or hiring decisions.
- The explanation bars describe how the **model** works, not what truly causes placement.

## Privacy

The app does not save anything to a server or database. Your inputs, uploaded files and history exist only in your browser session and disappear when you close the tab.

---

## What I learned

- Why input data must be **scaled the same way** in training and in the app
- How logistic regression turns numbers into a probability
- How to test a model honestly (a test split, plus cross-validation) and why a small dataset limits what you can trust
- How to turn a trained model into a complete, usable web app and put it online

---

## Author

Built by **Ayush**

- GitHub: https://github.com/ayushdere96-glitch
- LinkedIn: https://www.linkedin.com/in/ayush-dere-661907308
