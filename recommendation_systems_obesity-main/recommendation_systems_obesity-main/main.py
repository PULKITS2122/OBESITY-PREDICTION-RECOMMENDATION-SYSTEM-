import sys
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont, QIntValidator
from PyQt6.QtWidgets import QApplication, QTabWidget, QWidget, QLineEdit, QLabel, QPushButton
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.preprocessing import StandardScaler

# Read the dataset
data = pd.read_csv('obesity-final-edited.csv')

# Data preprocessing
X = data.drop(columns='NObeyesdad', axis=1)
Y = data['NObeyesdad']

# Removing rows where target variable 'Y' is NaN
valid_indices = Y.dropna().index
X = X.loc[valid_indices]
Y = Y.loc[valid_indices]

# One-hot encode categorical variables and handle missing values
X = pd.get_dummies(X, drop_first=True)
X.fillna(X.mean(), inplace=True)

# Scaling features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.25, random_state=2, stratify=Y)

# Train the model with increased max_iter
model = RandomForestClassifier(n_estimators=2112)
model.fit(x_train, y_train)

# Evaluate the model
predictions = model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))

class Window(FigureCanvas):
    def __init__(self, parent, data):
        fig, self.ax = plt.subplots(2, figsize=(8.9, 10))
        super().__init__(fig)
        self.setParent(parent)
        self.data = data

        # Initial plot setup
        self.update_plots(None, None)

    def update_plots(self, Age, Gender):
        # Clear existing plots
        self.ax[0].clear()
        self.ax[1].clear()

        # Update scatter plot (Age vs Gender)
        if Age is not None:
            filtered_data = self.data[self.data['Age'] == Age]
        else:
            filtered_data = self.data

        if Gender is not None:
            filtered_data = filtered_data[filtered_data['Gender'] == Gender]

        self.ax[0].scatter(filtered_data['Age'], filtered_data['Gender'])
        self.ax[0].set_xlabel('Age')
        self.ax[0].set_ylabel('Gender')
        self.ax[0].set_title('Scatter Plot of Age vs Gender')

        # Update line plot (Age vs Average Disease Classification)
        if Age is not None:
            line_plot_data = self.data.groupby('Age')['NObeyesdad'].mean().loc[:Age]
        else:
            line_plot_data = self.data.groupby('Age')['NObeyesdad'].mean()

        line_plot_data.plot(kind='line', ax=self.ax[1])
        self.ax[1].set_title('Age vs Average Obesity Classification')
        self.ax[1].set_xlabel('Age')
        self.ax[1].set_ylabel('Average Obesity Classification')

        # Refresh the plot
        self.draw()

    def save_current_plot(self, filename='plot_image.png'):
        self.figure.savefig(filename)

class AppMain(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.resize(900, 900)
        self.plot_window = Window(self, data)

def get_input_value(input_field):
    try:
        return int(input_field.text())  # Use .text() to get the string value
    except ValueError:
        return None

class MainWindow(QTabWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Obesity Prediction")
        self.setFixedSize(QSize(1500, 1000))
        self.font = QFont("Arial", 10)

        self.head = QLabel("Please Enter Patient's details.", self)
        self.head.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.head.move(900, 100)
        self.head.setAlignment(Qt.AlignmentFlag.AlignCenter)  # Changed for PyQt6

        self.label_Gender = QLabel("Gender (male:1 and female:0):", self)
        self.label_Gender.move(900, 220)
        self.label_Gender.setFont(self.font)
        self.value_Gender = QLineEdit(self)
        self.value_Gender.setValidator(QIntValidator(0,1, self))
        self.value_Gender.move(1100, 215)
        self.value_Gender.resize(150, 30)

        self.label_Age = QLabel("Age:", self)
        self.label_Age.move(900, 280)
        self.label_Age.setFont(self.font)
        self.value_Age = QLineEdit(self)
        self.value_Age.setValidator(QIntValidator(13, 61, self))
        self.value_Age.move(1100, 275)
        self.value_Age.resize(150, 30)


        self.label_family_history_with_overweight = QLabel("Family History:", self)
        self.label_family_history_with_overweight.move(900, 340)
        self.label_family_history_with_overweight.setFont(self.font)
        self.value_family_history_with_overweight = QLineEdit(self)
        self.value_family_history_with_overweight.setValidator(QIntValidator(0, 1, self))
        self.value_family_history_with_overweight.move(1100, 340)
        self.value_family_history_with_overweight.resize(150, 30)

        self.label_FAVC = QLabel("FAVC:", self)
        self.label_FAVC.move(900, 400)
        self.label_FAVC.setFont(self.font)
        self.value_FAVC = QLineEdit(self)
        self.value_FAVC.setValidator(QIntValidator(0, 1, self))
        self.value_FAVC.move(1100, 395)
        self.value_FAVC.resize(150, 30)

        self.label_FCVC = QLabel("FCVC:", self)
        self.label_FCVC.move(900, 460)
        self.label_FCVC.setFont(self.font)
        self.value_FCVC = QLineEdit(self)
        self.value_FCVC.setValidator(QIntValidator(1, 3, self))
        self.value_FCVC.move(1100, 460)
        self.value_FCVC.resize(150, 30)

        self.label_NCP = QLabel("NCP:", self)
        self.label_NCP.move(900, 520)
        self.label_NCP.setFont(self.font)
        self.value_NCP = QLineEdit(self)
        self.value_NCP.setValidator(QIntValidator(1, 4, self))
        self.value_NCP.move(1100, 520)
        self.value_NCP.resize(150, 30)

        self.label_CAEC = QLabel("CAEC:", self)
        self.label_CAEC.move(900, 580)
        self.label_CAEC.setFont(self.font)
        self.value_CAEC = QLineEdit(self)
        self.value_CAEC.setValidator(QIntValidator(0, 3, self))
        self.value_CAEC.move(1100, 580)
        self.value_CAEC.resize(150, 30)

        self.label_SMOKE = QLabel("SMOKE:", self)
        self.label_SMOKE.move(900, 640)
        self.label_SMOKE.setFont(self.font)
        self.value_SMOKE = QLineEdit(self)
        self.value_SMOKE.setValidator(QIntValidator(0, 1, self))
        self.value_SMOKE.move(1100, 640)
        self.value_SMOKE.resize(150, 30)

        self.label_CH2O = QLabel("CH2O:", self)
        self.label_CH2O.move(900, 700)
        self.label_CH2O.setFont(self.font)
        self.value_CH2O = QLineEdit(self)
        self.value_CH2O.setValidator(QIntValidator(1, 3, self))
        self.value_CH2O.move(1100, 700)
        self.value_CH2O.resize(150, 30)

        # Initialize the plot window as an attribute of MainWindow
        self.plot_window = Window(self, data)

        # Connect input change events for QLineEdit to update function
        self.value_Age.textChanged.connect(self.update_plots_and_prediction)
        self.value_Gender.textChanged.connect(self.update_plots_and_prediction)
        self.value_family_history_with_overweight.textChanged.connect(self.update_plots_and_prediction)
        self.value_FAVC.textChanged.connect(self.update_plots_and_prediction)
        self.value_FCVC.textChanged.connect(self.update_plots_and_prediction)
        self.value_NCP.textChanged.connect(self.update_plots_and_prediction)
        self.value_CAEC.textChanged.connect(self.update_plots_and_prediction)
        self.value_SMOKE.textChanged.connect(self.update_plots_and_prediction)
        self.value_CH2O.textChanged.connect(self.update_plots_and_prediction)

        # submit button
        self.button = QPushButton("Submit", self)
        self.button.setCheckable(True)
        self.button.move(900, 770)
        self.button.setCheckable(True)
        self.button.clicked.connect(self.predict_disease)  # Signal-slot connection remains the same

        # output label
        self.prediction = QLabel("", self)
        self.prediction.setGeometry(900, 720, 300, 50)
        self.head.setFont(QFont("Arial", 12, QFont.Weight.Bold))

        # Save image button
        self.save_image_button = QPushButton("Save Plot", self)
        self.save_image_button.setCheckable(True)
        self.save_image_button.move(980, 770)
        self.save_image_button.setCheckable(True)
        self.save_image_button.clicked.connect(self.save_image)

        main = AppMain()
        self.tab = QWidget(self)
        self.addTab(main, "Prediction Model")

    def predict_disease(self):
        try:
            # Check if the input fields are not empty
            if not all(
                    [self.value_Age.text(), self.value_Gender.text(), self.value_family_history_with_overweight.text(),
                     self.value_FAVC.text(), self.value_FCVC.text(), self.value_NCP.text(),
                     self.value_CAEC.text(), self.value_SMOKE.text(), self.value_CH2O.text()]):
                raise ValueError("Please fill in all fields")

            # Convert input to appropriate data types
            Age = int(self.value_Age.text())
            Gender = int(self.value_Gender.text())
            family_history_with_overweight = int(self.value_family_history_with_overweight.text())
            FAVC = int(self.value_FAVC.text())
            FCVC = int(self.value_FCVC.text())
            NCP = int(self.value_NCP.text())
            CAEC = int(self.value_CAEC.text())
            SMOKE = int(self.value_SMOKE.text())
            CH2O = int(self.value_CH2O.text())

            # Create a DataFrame for user input
            user_input_temp = pd.DataFrame(
                [[Gender,Age, family_history_with_overweight, FAVC, FCVC, NCP, CAEC, SMOKE, CH2O]],
                columns=['gender','age', 'family_history_with_overweight', 'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE',
                         'CH2O'])

            # Print the processed user input
            print("Processed user input:")
            print(user_input_temp)

            # Process user input
            user_input_encoded = pd.get_dummies(user_input_temp, drop_first=True)
            for col in X.columns:
                if col not in user_input_encoded.columns:
                    user_input_encoded[col] = 0
            user_input_encoded = user_input_encoded[X.columns]
            user_input_scaled = scaler.transform(user_input_encoded)

            # Make prediction
            prediction = model.predict(user_input_scaled)
            self.prediction.setText("Patient has Obesity." if prediction[0] == 1 else "Patient doesn't have Obesity.")

        except ValueError as ve:
            self.prediction.setText(f"Input Error: {ve}")
        except Exception as e:
            self.prediction.setText(f"An error occurred: {e}")

    def update_plots_and_prediction(self):
        # Extract current values from input fields
        Age = get_input_value(self.value_Age)
        Gender = get_input_value(self.value_Gender)

        # Update plots
        self.plot_window.update_plots(Age, Gender)

    def save_image(self):
        self.plot_window.save_current_plot()

# Main application execution
if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec())
