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
	 'totalzs_percentage']] #all float
## zs_weight_age, zs_height_age, zs_weight_height: Z-score gizi berdasarkan Age, Gender, Weight, Height 
(-3 < zscore < +3)
## totalzs_3: rata-rata dari tiga nilai zs 
(-3 < mean total zscore(mtz) < +3) (Kurang Gizi = mtz < -1 (misal=-1.1) , Obesitas = mtz > +1 (misal=+1.3), Gizi normal = -1 < mtz < +1 (misal: +0.6)
## totalzs_percentage: hasil transform nilai mtz ke persentase(%)
(100% < zscore < 100%)
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
