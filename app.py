import PIL
import numpy as np
import streamlit as st
import tflite_runtime.interpreter as tflite

max_length = 499
MODEL_TFLITE = 'model.tflite'
label_dict = {2:'Positive', 1:'Neutral', 0:'Negative'}
word_to_index = np.load('weights/word_to_index.npy', allow_pickle=True).item()

def pad_sequences(sequences, maxlen = None, dtype = 'int32', padding = 'pre', truncating = 'pre', value = 0.):

    num_samples = len(sequences)

    lengths = []
    sample_shape = ()
    flag = True

    for x in sequences:
        try:
            lengths.append(len(x))
            if flag and len(x):
                sample_shape = np.asarray(x).shape[1:]
                flag = False
        except TypeError:
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))

    if maxlen is None:
        maxlen = np.max(lengths)

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)


    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x

def predict(sentence):
    X_new = np.array([word_to_index[w] for w in sentence.split() if w in word_to_index])
    X_new = X_new.reshape(1, X_new.shape[0])
    seq = pad_sequences(X_new, maxlen = max_length, padding = 'post', truncating = 'post')
    seq = np.float32(seq)

    interpreter = tflite.Interpreter(model_path = MODEL_TFLITE)
    interpreter.allocate_tensors()

    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']

    interpreter.set_tensor(input_index, seq)
    interpreter.invoke()

    pred = np.argmax(interpreter.get_tensor(output_index))
    pred = label_dict[pred]

    return pred

st.title('Amazon Fine Food Review')
st.write(":heavy_minus_sign:" * 30)

# Sidebar
with st.sidebar:
    st.title('About the Fine Food Dataset')
    st.write('Fine foods are usually products of great quality from around the world. Usually, the rich and aristocratic people prefer fine foods. The **Amazon Fine Food Dataset** is taken from Kaggle. It had around **500,000** food reviews upto October 2012.')

    st.title('Why TensorFlow-Lite?')
    st.markdown('I have used **TensorFlow** to build the model and used **TensorFlow-Lite** for inference. I use TF-Lite because the size of TensorFlow model was **156 MB**, whereas the size of TF-Lite model is **42 MB** with the same accuracy! Also the TensorFlow Package cost **~450MB**, which results in increased Docker Image Size. So, I got **rid** of TensorFlow dependencies and used just **2.5 MB** TF-Lite Runtime Package.')


# TATA Tea
tea_image = PIL.Image.open('images/tata_tea.jpg')
tea_image = tea_image.resize((350, 350))

# Cadbury
cadbury = PIL.Image.open('images/cadbury.jpg')
cadbury = cadbury.resize((350, 350))

# Nescafe
nescafe = PIL.Image.open('images/nescafe.jpg')
nescafe = nescafe.resize((350, 350))

# Indian Sweet
sweet = PIL.Image.open('images/sweet.jpg')
sweet = sweet.resize((350, 350))

tab1, tab2, tab3, tab4 = st.tabs(['TATA Tea', 'Cadbury', 'Nescafe Gold', 'Indian Sweets'])
tab1.write('Imagine you have brought a packet of TATA Tea from Amazon. Please give a review below.')
tab1.image(tea_image)
tab1_review = tab1.text_input('Enter your review here ')
if tab1_review:
    tab1_sentiment = predict(tab1_review)
    tab1.write(tab1_sentiment)

tab2.write('Imagine you have brought a packet of Cadbury Dry Fruits from Amazon. Please give a review below.')
tab2.image(cadbury)
tab2_review = tab2.text_input('Enter your review here  ')
if tab2_review:
    tab2_sentiment = predict(tab2_review)
    tab2.write(tab2_sentiment)

tab3.write('Imagine you have brought a packet of Nescafe Gold from Amazon. Please give a review below.')
tab3.image(nescafe)
tab3_review = tab3.text_input('Enter your review here   ')
if tab3_review:
    tab3_sentiment = predict(tab3_review)
    tab3.write(tab3_sentiment)

tab4.write('Imagine you have brought a canned Gulab Jamun from Amazon. Please give a review below.')
tab4.image(sweet)
tab4_review = tab4.text_input('Enter your review here    ')
if tab4_review:
    tab4_sentiment = predict(tab4_review)
    tab4.write(tab4_sentiment)