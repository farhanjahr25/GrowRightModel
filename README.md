# Fitur
X = data[['gender (string: male, female)',
	 'age(int: 0-60 bulan)',
	 'weight(float: kg)',
	 'height(float: cm)']]
# Target
y = data[['zs_weight_age',
	 'zs_height_age',
	 'zs_weight_height',
     'totalzs_3',
	 'totalzs_percentage']] #float
# def GenderEncode
def genderEncode(gender, age, weight, height):

    # Label encoding gender
    gender_encoding = {'male': 1, 'female': 0}
    # Encode gender
    gender_encoded = gender_encoding.get(gender)

    return gender_encoded, age, weight, height
# def normalize_data with MinMaxScaler()
def normalize_data(gender_encoded, age, weight, height):

    normalization_model = "normalization_model.joblib"
    # Load the saved scaler using joblib
    scaler = joblib.load(normalization_model)
    feature = np.array([[gender_encoded, age, weight, height]])
    # Normalize the data using Joblib
    normalized_data = scaler.transform(feature)

    return normalized_data
