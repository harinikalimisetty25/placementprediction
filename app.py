from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import os

app = Flask(__name__)

# ============================================================
# UPLOAD FOLDER
# ============================================================

UPLOAD_FOLDER = "uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create uploads folder automatically
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ============================================================
# GLOBAL DATASET
# ============================================================

dataset_df = None


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():
    return render_template("home.html")


# ============================================================
# ABOUT
# ============================================================

@app.route("/about")
def about():
    return render_template("about.html")


# ============================================================
# DATASET PAGE
# ============================================================

@app.route("/dataset")
def dataset():

    global dataset_df

    if dataset_df is not None:

        total_students = len(dataset_df)

        total_features = len(dataset_df.columns)

        missing_values = int(
            dataset_df.isnull().sum().sum()
        )

        duplicate_records = int(
            dataset_df.duplicated().sum()
        )

    else:

        total_students = 0
        total_features = 0
        missing_values = 0
        duplicate_records = 0

    return render_template(
        "dataset.html",
        total_students=total_students,
        total_features=total_features,
        missing_values=missing_values,
        duplicate_records=duplicate_records
    )


# ============================================================
# UPLOAD DATASET
# ============================================================

@app.route("/upload-dataset", methods=["POST"])
def upload_dataset():

    global dataset_df

    # Get uploaded file
    file = request.files.get("dataset")

    # Check file
    if file is None or file.filename == "":

        return render_template(
            "dataset.html",
            total_students=0,
            total_features=0,
            missing_values=0,
            duplicate_records=0,
            message="Please select a CSV file first."
        )

    # Check CSV extension
    if not file.filename.lower().endswith(".csv"):

        return render_template(
            "dataset.html",
            total_students=0,
            total_features=0,
            missing_values=0,
            duplicate_records=0,
            message="Only CSV files are allowed."
        )

    try:

        # Save file
        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            file.filename
        )

        file.save(filepath)

        # Read CSV
        dataset_df = pd.read_csv(filepath)

        print("\n======================================")
        print("DATASET UPLOADED SUCCESSFULLY")
        print("======================================")

        print("File Name:", file.filename)

        print("Dataset Shape:", dataset_df.shape)

        print("Total Rows:", len(dataset_df))

        print("Total Columns:", len(dataset_df.columns))

        print(
            "Missing Values:",
            dataset_df.isnull().sum().sum()
        )

        print(
            "Duplicate Records:",
            dataset_df.duplicated().sum()
        )

        print("======================================\n")

        # Return to dataset page
        return redirect(url_for("dataset"))

    except Exception as e:

        return render_template(
            "dataset.html",
            total_students=0,
            total_features=0,
            missing_values=0,
            duplicate_records=0,
            message="Error loading dataset: " + str(e)
        )


# ============================================================
# VIEW DATASET
# ============================================================

@app.route("/view-dataset")
def view_dataset():

    global dataset_df

    # Check whether dataset exists
    if dataset_df is None:

        return render_template(
            "view_dataset.html",
            message="No dataset uploaded. Please upload a CSV dataset first.",
            table=None,
            total_rows=0
        )

    try:

        # Display first 100 rows
        data = dataset_df.head(100)

        # Convert dataframe into HTML table
        table = data.to_html(
            classes="data-table",
            index=False,
            border=0
        )

        return render_template(
            "view_dataset.html",
            message=None,
            table=table,
            total_rows=len(dataset_df)
        )

    except Exception as e:

        return render_template(
            "view_dataset.html",
            message="Error displaying dataset: " + str(e),
            table=None,
            total_rows=0
        )


# ============================================================
# DATASET SUMMARY
# ============================================================

@app.route("/dataset-summary")
def dataset_summary():

    global dataset_df

    # Check whether dataset exists
    if dataset_df is None:

        return render_template(
            "summary.html",
            message="No dataset uploaded. Please upload a CSV dataset first.",
            summary=None
        )

    try:

        # Generate statistical summary
        summary = dataset_df.describe(
            include="all"
        ).transpose()

        # Convert to HTML
        summary_table = summary.to_html(
            classes="data-table",
            border=0
        )

        return render_template(
            "summary.html",
            message=None,
            summary=summary_table
        )

    except Exception as e:

        return render_template(
            "summary.html",
            message="Error generating summary: " + str(e),
            summary=None
        )


# ============================================================
# PREPROCESSING
# ============================================================

@app.route("/preprocessing")
def preprocessing():
    return render_template("preprocessing.html")


# ============================================================
# VISUALIZATION
# ============================================================

@app.route("/visualization")
def visualization():
    return render_template("visualization.html")


# ============================================================
# ML MODELS
# ============================================================

@app.route("/models")
def models():
    return render_template("models.html")


# ============================================================
# PREDICTION
# ============================================================

@app.route("/prediction")
def prediction():
    return render_template("prediction.html")


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


# ============================================================
# REPORTS
# ============================================================

@app.route("/reports")
def reports():
    return render_template("reports.html")


# ============================================================
# CONTACT
# ============================================================

@app.route("/contact")
def contact():
    return render_template("contact.html")


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)