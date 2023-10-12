import numpy as np
import pickle
import streamlit as st

# loading the saved model
load_model = pickle.load(open('D:/Cipherbyte_Tech/Task 2/SpamEmailDetection/savemodel.sav', 'rb'))

# creating a function for prediction
def spam_Email_Detection(input_data):
    input_data = ("Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's")
    #changing the input data to numpy array
    input_data_as_numpy_array = np.array(input_data)
    #reshaping the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1)
    prediction = load_model.predict(input_data_reshaped)
    print(prediction)

    if(prediction[0] == 'ham'):
        return 'ham'
    elif(prediction[0] == 'spam'):
        return 'spam'

    

def main():
    # giving a title
    st.title('Spam Email Detection Web App')

    # getting the input data from the user
    v2 = st.text_input("Enter the text")


    # Code for Prediction
    v1 = ''

    # creating a button for Prediction
    if st.button('Result'):
        # species = spam_Email_Detection([v2])
        prediction_label = spam_Email_Detection([v2])
        st.write('Prediction:', prediction_label)


    st.success(v1)

if __name__ == '__main__':
    main()


















