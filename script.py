#!/usr/bin/env python
# coding: utf-8

# # MétéoSensiblité
# ## Objectif: l'analyse de la météo-sensibilité des produits (faire la prévision des ventes en fonction de la météo)

# Même si les données ne présentent pas de défauts particuliers, une étude exploratoire préliminaire est indispensable afin de s'assurer leur bonne cohérence, proposer d'éventuelles transformations et analyser les structures de corrélations ou plus généralement de liaisons entre les variables, de groupes des individus ou observations.
# 
# 
# # 1. Analyse exploratoire de données & PRÉ-TRAITEMENT DE DONNÉES
# 
# ## Objectif :
# 
# - Préparation des données : Obtenir une vision globale d’un jeu de données
# - Comprendre du mieux possible nos données (un petit pas en avant vaut mieux qu'un grand pas en arriere)
# - Développer une premiere stratégie de modélisation 
# 
# ## Checklist de base
# 
# #### Préparation des données : 
# - **Importation de données** 
# - **Analyse des valeurs manquantes** 
# - **Les jointures entre tables (Data Produits & Bases logistiques & Méteo)** 
# 
# #### Analyse de Forme :
# - **variable target** : Qtés Vendues produit
# - **lignes et colonnes** : 278707, 16
# - **Analyse des valeurs manquantes** : Pas de valeurs manquantes 
# 
# #### Analyse de Fond :
# - **Visualisation de la target** 
#    
# ## Analyse plus détaillée
# 
# - **Relation Variables / Variables** :
#     - Qtés Vendues produit/ tmin, tmax, tmoy : certaines variables sont tres corrélées : +0.8 
#     - On remarque une liaison linéaire positive. tmin, tmax, tmoy et Qtés Vendues produit évoluent dans le même sens, une augmentation de tmin/tmax/tmoy entraîne une augmentation de Qtés Vendues produit , du même ordre quelle que soit la valeur de tmin/tmax/tmoy.
# ## La détection des outliers :  
#     - la suppression des valeaurs extrêmes ( Quantité vendus < 0 ) 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


# In[2]:


df_produit=pd.read_excel('Bières 2015-2017.xlsx') 
df_produit=df_produit.loc[(df_produit['Qtés Vendues produit'] >= 0),:]
df_produit.head()


# In[3]:


plt.figure(figsize=(10,5))
sns.heatmap(df_produit.isna(), cbar=False)


# In[4]:


df_produit=df_produit.fillna(axis=0, method='ffill')
plt.figure(figsize=(10,5))
sns.heatmap(df_produit.isna(), cbar=False)


# In[5]:


df_produit=df_produit.sort_index(ascending=False)
df_produit.drop([0], inplace=True)
df_produit


# In[6]:


#baseslog=pd.read_excel('pdv.xlsx')
#baseslog=pd.read_excel('pdv_modif.xlsx')
baseslog=pd.read_excel('pdv_modiff.xlsx')
baseslog


# In[7]:


baseslog.rename(columns={'id': 'Point de Vente'}, inplace=True)
df_produit1= pd.merge(df_produit,
                 baseslog[['cp','id_base_log_sec','id_base_log_frais','Point de Vente']],
                 on='Point de Vente', 
                 how='left')
df_produit1.describe()


# In[8]:


df_produit1=df_produit1.dropna()
df_produit1.describe()


# In[9]:


df_produit2=df_produit1.join(df_produit1['Semaine'].str.split('-', 1, expand=True).rename(columns={0:'year', 1:'semaine'}))
df_produit2


# In[10]:


df_produit3=df_produit2.join(df_produit2['semaine'].str.extract('(\d+)').astype(int).rename(columns={0:'week'}))
df_produit3=df_produit3.drop(['semaine','Semaine'],axis=1)
df_produit3["year"] = df_produit3["year"].astype(str).astype(int)
df_produit3["cp"] = df_produit3["cp"].astype(float).astype(int)
df_produit3["Point de Vente"] = df_produit3["Point de Vente"].astype(float).astype(int)
df_produit3["id_base_log_sec"] = df_produit3["id_base_log_sec"].astype(float).astype(int)
df_produit3


# In[11]:


meteo=pd.read_csv("METEO.csv")
meteo.rename(columns={'annee': 'year', 'semaine':'week'}, inplace=True)
meteo


# In[12]:


meteo_produit= pd.merge(df_produit3,
                 meteo,
                 on=['cp','year','week'],
                 how='left')
meteo_produit.describe()
def year_week(y, w):
    return datetime.strptime(f'{y} {w} 1', '%G %V %u')
meteo_produit['Date'] = meteo_produit.apply(lambda row: year_week(row.year, row.week), axis=1)
meteo_produit


# In[13]:


meteo_produit=meteo_produit[meteo_produit['Point de Vente'] == 1001]
meteo_produit=meteo_produit.drop(['Name','year'],axis=1)


# In[14]:


meteo_produit.rename(columns={'Qtés Vendues produit':'Quantités_vendues'}, inplace=True)


# In[15]:


meteo_produit['Date'] = pd.to_datetime(meteo_produit['Date'])
meteo_produit = meteo_produit.set_index('Date')
meteo_produit


# # la structure de corrélation des variables 

# In[16]:


matrice_corr =meteo_produit.corr().round(1)
sns.heatmap(data=matrice_corr, annot=True)


# In[17]:


#sns.pairplot(meteo_produit)


# In[18]:


df=meteo_produit[['Quantités_vendues','tmoy','tmax','tmin','week','apcp','tcdc']]
#Plot
fig, axes = plt.subplots(nrows=2, ncols=2, dpi=120, figsize=(10,6))
for i, ax in enumerate(axes.flatten()):
    data = df[df.columns[i]]
    ax.plot(data, color='red', linewidth=1)
    # Decorations
    ax.set_title(df.columns[i])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)

plt.tight_layout();


# In[19]:


#sns.catplot(x='tmoy', y='Quantités_vendues', data=meteo_produit, hue='tmoy')
#sns.boxplot(x='tmoy', y='Quantités_vendues', data=meteo_produit )
#sns.distplot(meteo_produit['Quantités_vendues'])
meteo_produit.plot.scatter(x='tmoy',y='Quantités_vendues',color='red')
meteo_produit.plot.scatter(x='tmax',y='Quantités_vendues')


# - On remarque que la courbe de ventes de bières suivait scrupuleusement celle des températures : augmentation des ventes dès 20 degrés
# - tmin, tmax, tmoy et Quantités_vendues évoluent dans le même sens, une augmentation de tmin/tmax/tmoy entraîne une augmentation de la quantité vendu de bières 

# # 2. Check for Stationarity and Make the Time Series Stationary

# Most time-series models assume that the underlying time-series data is **stationary**.  This assumption gives us some nice statistical properties that allows us to use various models for forecasting.
# 
# **Stationarity** is a statistical assumption that a time-series has:
# *   **Constant mean**
# *   **Constant variance**
# *   **Autocovariance does not depend on time**
# 
# More simply put, if we are using past data to predict future data, we should assume that the data will follow the same general trends and patterns as in the past.  This general statement holds for most training data and modeling tasks.
# 
# **There are some good diagrams and explanations on stationarity [here](https://www.analyticsvidhya.com/blog/2015/12/complete-tutorial-time-series-modeling/) and [here](https://people.duke.edu/~rnau/411diff.htm).**
# 
# Sometimes we need to transform the data in order to make it stationary.  However, this  transformation then calls into question if this data is truly stationary and is suited to be modeled using these techniques.
# 
# **Looking at our data:**
# - Rolling mean and standard deviation look like they change over time.  There may be some de-trending and removing seasonality involved. Based on **Dickey-Fuller test**, because p = 0.31, we fail to reject the null hypothesis (that the time series is not stationary) at the p = 0.05 level, thus concluding that we fail to reject the null hypothesis that our **time series is not stationary**.

# In[20]:


#Stationarity is a statistical assumption that a time-series has: Constant mean, Constant variance, Autocovariance does not depend on time
#Test stationarity using moving average statistics and Dickey-Fuller test
from statsmodels.tsa.stattools import adfuller
def test_stationarity(df, Quantités_vendues):        
    # Determing rolling statistics
    rolmean = df[Quantités_vendues].rolling(window = 12, center = False).mean()
    rolstd = df[Quantités_vendues].rolling(window = 12, center = False).std()
    
    # Plot rolling statistics:
    orig = plt.plot(df[Quantités_vendues], 
                    color = 'blue', 
                    label = 'Original')
    mean = plt.plot(rolmean, 
                    color = 'red', 
                    label = 'Rolling Mean')
    std = plt.plot(rolstd, 
                   color = 'black', 
                   label = 'Rolling Std')
    plt.legend(loc = 'best')
    plt.title('Rolling Mean & Standard Deviation for %s' %(Quantités_vendues))
    plt.xticks(rotation = 45)
    plt.show(block = False)
    plt.close()    
    # Perform Dickey-Fuller test:
    # Null Hypothesis (H_0): time series is not stationary
    # Alternate Hypothesis (H_1): time series is stationary
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(df[Quantités_vendues], 
                      autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], 
                         index = ['Test Statistic',
                                  'p-value',
                                  '# Lags Used',
                                  'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    return dfoutput


# In[21]:


test_stationarity(df, Quantités_vendues='Quantités_vendues')


# In[22]:


from statsmodels.tsa.stattools import adfuller
def adfuller_test(series, signif=0.05, name='', verbose=False):
    """Perform ADFuller to test for Stationarity of given series and print report"""
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue'] 
    def adjust(val, length= 6): return str(val).ljust(length)

    # Print Summary
    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
    print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
    print(f' Significance Level    = {signif}')
    print(f' Test Statistic        = {output["test_statistic"]}')
    print(f' No. Lags Chosen       = {output["n_lags"]}')

    for key,val in r[4].items():
        print(f' Critical value {adjust(key)} = {round(val, 3)}')

    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series is Stationary.")
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is Non-Stationary.")    


# In[23]:


# ADF Test on each column
for name, column in df.iteritems():
    adfuller_test(column, name=column.name)
    print('\n')


# In[24]:


# 1st difference
#The ADF test confirms none of the time series is stationary Let’s difference all of them once and check again.
#df_differenced = df.diff().dropna()
#Or, either proceed with 1st differenced series or difference all the series one more time.
# Second Differencing
#df_differenced = df_differenced.diff().dropna()


# # 3. Modélisation
# la recherche d'une meilleure méthode de prévision suit généralement le protocole suivant dont la première étape est déja réalisée.
# 

# #  1.Facebook Prophet package.
# 
# We will be doing an example here! Installing the necessary packages might take a couple of minutes.  In the meantime, I can talk a bit about [Facebook Prophet](https://facebook.github.io/prophet/), a tool that allows folks to forecast using additive or component models relatively easily.  It can also include things like:
# * Day of week effects
# * Day of year effects
# * Holiday effects
# * Trend trajectory
# * Can do MCMC sampling

# In[25]:


get_ipython().system('pip install pystan')
get_ipython().system('pip install fbprophet')
from fbprophet import Prophet
import datetime
from datetime import datetime


# In[30]:


df_météoSensibilité =  df.reset_index()[['Date', 'Quantités_vendues', 'week']].rename({'Date':'ds',
                                                                          'Quantités_vendues':'y', 
                                                                          'week':'week' }, axis='columns')


# In[31]:


df_météoSensibilité


# In[32]:


train=df_météoSensibilité[(df_météoSensibilité['ds']< '2017-06-7')]
test=df_météoSensibilité[(df_météoSensibilité['ds']>='2017-06-7')]
threshold_date = pd.to_datetime('2017-06-7')


# In[33]:


train.shape


# In[34]:


test.shape


# In[35]:


test.tail()


# In[36]:


fig, ax = plt.subplots(figsize=(12,5))
sns.lineplot(x='ds', y='y', label='y_train', data=train, ax=ax)
sns.lineplot(x='ds', y='y', label='y_test', data=test, ax=ax)
ax.axvline(threshold_date,linestyle='--', label='train test split')
ax.legend(loc='upper left')
ax.set(title='Dependent Variable', ylabel='');


# In[37]:


model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False, 
        interval_width=0.95, 
        mcmc_samples = 500
    )


# In[38]:


#model.add_regressor('tmoy', standardize=False)
#model.add_regressor('tmax', standardize=False)
model.add_regressor('week', standardize=False)
#model.add_regressor('tmin', standardize=False)


# In[39]:


model.fit(train)


# In[40]:


model.params


# In[41]:


future = model.make_future_dataframe(periods=test.shape[0], freq='W')
future.tail()


# In[42]:


#future['tmoy'] = df_météoSensibilité['tmoy']
#future['tmin'] = df_météoSensibilité['tmin']
#future['tmax'] = df_météoSensibilité['tmax']
future['week'] = df_météoSensibilité['week']


# In[43]:


future


# In[44]:


forecast =model.predict(future)
forecast[['ds','yhat','yhat_lower','yhat_upper']].tail()


# In[45]:


fig1 = model.plot(forecast,figsize=(10,6))


# In[46]:


fig2 = model.plot_components(forecast)


# In[47]:


#Let us split the predictions into training and test set.
fig, ax = plt.subplots(figsize=(10,5))

ax.fill_between(
    x=forecast['ds'],
    y1=forecast['yhat_lower'],
    y2=forecast['yhat_upper'],
    alpha=0.25,
    label=r'0.95 credible_interval'
)

sns.lineplot(x='ds', y='y', label='y_train', data=train, ax=ax)
sns.lineplot(x='ds', y='y', label='y_test', data=test, ax=ax)
sns.lineplot(x='ds', y='yhat', label='y_hat', data=forecast, ax=ax)
ax.axvline(threshold_date, linestyle='--', label='train test split')
ax.legend(loc='upper left')
ax.set(title='Dependent Variable', ylabel='');


# In[48]:


A = forecast['ds'] < threshold_date
forecast_train = forecast[A]
forecast_test = forecast[~ A]


# In[49]:


fig, ax = plt.subplots(figsize=(10,5))

ax.fill_between(
    x=forecast_test['ds'],
    y1=forecast_test['yhat_lower'],
    y2=forecast_test['yhat_upper'],
    alpha=0.25,
    label=r'0.95 credible_interval'
)
sns.lineplot(x='ds', y='y', label='y_test', data=test, ax=ax)
sns.lineplot(x='ds', y='yhat', label='y_hat', data=forecast_test, ax=ax)
ax.legend(loc='lower right')
ax.set(title='Dependent Variable', ylabel='');


# In[50]:


fig, ax = plt.subplots(figsize=(8,8))
# Generate diagonal line to plot. 
d_x = np.linspace(start=train['y'].min() - 1, stop=train['y'].max() + 1, num=100)
sns.regplot(x=train['y'], y=forecast_train['yhat'],  label='train', ax=ax)
sns.regplot(x=test['y'], y=forecast_test['yhat'],  label='test', ax=ax)
sns.lineplot(x=d_x, y=d_x, dashes={'linestyle': ''},  ax=ax)
ax.lines[2].set_linestyle('--')
ax.set(title='Test Data vs Predictions');


# In[51]:


#Let us compute the r2_score and mean_absolute_error on the training and test set respectively:
from sklearn.metrics import r2_score, mean_absolute_error
def calculate_mape(y_true, y_pred):
    """ Calculate mean absolute percentage error (MAPE)"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def calculate_mpe(y_true, y_pred):
    """ Calculate mean percentage error (MPE)"""
    return np.mean((y_true - y_pred) / y_true) * 100

def calculate_mae(y_true, y_pred):
    """ Calculate mean absolute error (MAE)"""
    return np.mean(np.abs(y_true - y_pred)) * 100

def calculate_rmse(y_true, y_pred):
    """ Calculate root mean square error (RMSE)"""
    return np.sqrt(np.mean((y_true - y_pred)**2))

def print_error_metrics(y_true, y_pred):
    print('MAPE: %f'%calculate_mape(y_true, y_pred))
    print('MPE: %f'%calculate_mpe(y_true, y_pred))
    print('RMSE: %f'%calculate_rmse(y_true, y_pred))
    return
print_error_metrics(y_true = df_météoSensibilité['y'], y_pred = forecast['yhat'])
print('---'*10)
print('r2 train: {}'.format(r2_score(y_true=train['y'], y_pred=forecast_train['yhat'])))
print('r2 test: {}'.format(r2_score(y_true=test['y'], y_pred=forecast_test['yhat'])))
print('---'*10)
print('mae train: {}'.format(mean_absolute_error(y_true=train['y'], y_pred=forecast_train['yhat'])))
print('mae test: {}'.format(mean_absolute_error(y_true=test['y'], y_pred=forecast_test['yhat'])))


# In[52]:


#Error Analysis
#Let us study the forecast errors.
#1) Distribution
forecast_test.loc[:, 'errors'] = forecast_test.loc[:, 'yhat'] - test.loc[:, 'y']

errors_mean = forecast_test['errors'].mean()
errors_std = forecast_test['errors'].std()

fig, ax = plt.subplots()

sns.distplot(a=forecast_test['errors'], ax=ax, bins=15, rug=True)
ax.axvline(x=errors_mean,  linestyle='--', label=r'$\mu$')
ax.axvline(x=errors_mean + 2*errors_std,  linestyle='--', label=r'$\mu \pm 2\sigma$')
ax.axvline(x=errors_mean - 2*errors_std,  linestyle='--')
ax.legend()
ax.set(title='Model Errors (Test Set)');


# In[53]:


#2)Autocorrelation
fig, ax = plt.subplots(figsize=(12,6))
sns.scatterplot(x='index', y='errors', data=forecast_test.reset_index(), ax=ax)
ax.axhline(y=errors_mean,  linestyle='--', label=r'$\mu$ ')
ax.axhline(y=errors_mean + 2*errors_std,  linestyle='--', label=r'$\mu \pm 2\sigma$')
ax.axhline(y=errors_mean - 2*errors_std,  linestyle='--')
ax.legend()
ax.set(title='Model Errors (Test Set)');


# In[54]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
fig, ax = plt.subplots(2, 1,figsize=(8,7))
plot_acf(x=forecast_test['errors'], ax=ax[0])
plot_pacf(x=forecast_test['errors'], ax=ax[1]);


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




