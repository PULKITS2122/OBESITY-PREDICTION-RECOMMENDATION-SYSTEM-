import pandas as pd

# Load the dataset
df = pd.read_csv('obesity.csv')

# Open the description file in write mode
with open('description.txt', 'w') as f:
    # Write the description of the dataset
    f.write('Description of Dataset:\n')
    for column in df.columns:
        f.write(f'{column}: Description of {column}\n')

    # Write the variables ignored
    f.write('\nvariables ignored:\n')
    ignored_variables = ['SCC','FAF','TUE','CALC','Automobile','Bike','Motorbike','Public_Transportation','Walking'] # replace with your actual ignored variables
    for variable in ignored_variables:
        f.write(f'{variable}\n')


# Remove the specified columns
df = df.drop(columns=ignored_variables)

# Save the modified data to a new file
df.to_csv('obesity-final.csv', index=False)
