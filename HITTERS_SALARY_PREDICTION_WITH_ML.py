import pandas as pd
from sklearn.impute import SimpleImputer
from helpers.eda import *
from helpers.data_prep import *
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

import warnings
warnings.simplefilter(action='ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from warnings import filterwarnings
filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 170)

df_=pd.read_csv("Hafta7/hitters.csv")
df=df_.copy()
df.shape #322 gözlem birimi, 20 değişken var.
df.head()

df["Salary"].isnull().sum()  #Bağımlı değişkende 59 tane NaN değeri var.
df["Salary"].describe()
sns.distplot(df.Salary)
plt.show()
cat_cols, num_cols, cat_but_car = grab_col_names(df) #Kategorik,Sayısal,Kardinal değişkenleri görelim.
cat_cols  #['League':A N seviyeleri var,'Division':E ve W seviyeleri var.'NewLeague':A N seviyeleri var]
rare_analyser(df, "Salary", cat_cols) #['League', 'Division', 'NewLeague'] için bağımlı değişken ile beraber baktık.
for col in num_cols:
    num_summary(df, col, plot=False)
num_cols  #numerik değişikenleri görelim.
for col in num_cols:
    print(check_outlier(df,col))  #Aykırı gözlem var mı? yok çünkü False döndü hepsi.

# Eksik Gözlem var mı?
missing_values_table(df)
"""       n_miss  ratio
Salary      59 18.320"""

# Oyuncunun tecrübesinin önemli olduğunu düşünebilirz. Years değişkenine göre kırıp yeni değişken türetelim:
df["NEW_YEAR_CAT"] = pd.qcut(df["Years"], q=4)
#Salary değişkeninin NaN değerlerini kırılıma göre median ile dolduralım:
df["Salary"] = df["Salary"].fillna(df.groupby(["League", "Division", "NEW_YEAR_CAT"])["Salary"].transform("median"))
df["Salary"].isnull().sum()  #0 geldi eksik değer kalmamış.

df['At/CAt'] = df['AtBat'] / df['CAtBat'] #Topa vurma oranı değişkeni türettik.
df['Hits/CHits'] = df['Hits'] / df['CHits'] #Bu sezondaki isabetli atış oranını veren değişken türettik.
df['Runs/CRuns'] = df['Runs'] / df['CRuns'] #Bu sezondaki sayı kazandırma oranını veren değişken türettik.

# Yıl baz alınarak istatistikler:
df["NEW_AVG_ATBAT"] = df["CAtBat"] / df["Years"]
df["NEW_AVG_HITS"] = df["CHits"] / df["Years"]
df["NEW_AVG_HMRUN"] = df["CHmRun"] / df["Years"]
df["NEW_AVG_RUNS"] = df["CRuns"] / df["Years"]
df["NEW_AVG_RBI"] = df["CRBI"] / df["Years"]
df["NEW_AVG_WALKS"] = df["CWalks"] / df["Years"]

# Korelasyon Analizi yapalım:
"""Korelasyon katsayısı;
+1,00′ a yaklaştıkça iki değişken arasında aynı yöndeki ilişki artar.Değişkenlerden biri artarken diğeri de artar.
-1,00′ a yaklaştıkça iki değişen arasında ters yönde ilişki artar. Değişkenlerden biri artarken diğeri azalır.
0,00’a yaklaştıkça iki değişken arasındaki ilişki azalır."""
def target_correlation_matrix(dataframe, corr_th=0.7, target="Salary"):
    """
    Bağımlı değişken ile verilen threshold değerinin üzerindeki korelasyona sahip değişkenleri getirir.
    :param dataframe:
    :param corr_th: eşik değeri
    :param target:  bağımlı değişken ismi
    :return:
    """
    corr = dataframe.corr()
    corr_th = corr_th
    try:
        filter = np.abs(corr[target]) > corr_th
        corr_features = corr.columns[filter].tolist()
        sns.clustermap(dataframe[corr_features].corr(), annot=True, fmt=".2f")
        plt.show()
        return corr_features
    except:
        print("Yüksek threshold değeri, corr_th değerinizi düşürün!")
target_correlation_matrix(df, corr_th=0.5, target="Salary")
df.corr()

# Değişken türettikten sonra kateogirk ve numerik değişkenleri tekrar atamalıyız:
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Model kurmadan önce One-Hot encoder kategorik değişkenlerin temsil şeklini değiştirelim:
df = one_hot_encoder(df, cat_cols, drop_first=True)
df.head()

# MODEL:
y = df['Salary']
X = df.drop("Salary", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
reg = LinearRegression()
reg_model = reg.fit(X_train, y_train)

# TEST RMSE :Test hatamıza bakalım.
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred)) #291.9050581949059
y_pred[0:5] #İlk 5 gözlem: array([ 339.28180762,  714.53916592,   22.97591371,  879.9508692 ,1193.67806423])
















