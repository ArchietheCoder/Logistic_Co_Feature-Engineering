#!/usr/bin/env python
# coding: utf-8

# # <span style='color: brown;'>Business Case - Delhivery: Feature Engineering</span>

# ## <span style='color: blue;'>1. Problem Statement</span>

# ## About Delhivery
# Delhivery is the largest and fastest-growing fully integrated player in India by revenue in Fiscal 2021. They aim to build the operating system for commerce, through a combination of world-class infrastructure, logistics operations of the highest quality, and cutting-edge engineering and technology capabilities. The Data team builds intelligence and capabilities using this data that helps them to widen the gap between the quality, efficiency, and profitability of their business versus their competitors.
# 
# ## Objective of the Data Analysis?
# The company wants to understand and process the data coming out of data engineering pipelines:
#         
#         • Clean, sanitize and manipulate data to get useful features out of raw fields
# 
#         • Make sense out of the raw data and help the data science team to build forecasting models on it

# ### Importing Required Libraries

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as spy


# ### Loading the given data-set

# In[3]:


df = pd.read_csv('D:\\Scaler\\Scaler\\Hypothesis Testing\\Business Case\\delhivery_data.csv')


# In[4]:


df.head()


# ## <span style='color: blue;'>2. Data Exploration</span> 

# In[5]:


df.shape


# In[6]:


df.columns


# In[7]:


df.dtypes


# In[8]:


df.info()


# ### Dropping unknown fields

# In[9]:


unknown_fields = ['is_cutoff', 'cutoff_factor', 'cutoff_timestamp', 'factor', 'segment_factor']
df = df.drop(columns = unknown_fields)


# In[10]:


df.info()


# ## Identifying Category Columns

# ### In order to find the category columns, we will consider columns with 2 unique values

# In[11]:


for i in df.columns:
    print(f"Unique entries for column {i:<30} = {df[i].nunique()}")


# In[12]:


cat_cols = ['data', 'route_type']
for col in cat_cols:
    df[col] =  df[col].astype('object')


# In[13]:


floating_columns = ['actual_distance_to_destination', 'actual_time', 'osrm_time', 'osrm_distance', 
                    'segment_actual_time', 'segment_osrm_time', 'segment_osrm_distance', 'start_scan_to_end_scan']
for col in floating_columns:
    print(f"{col:<30} = {df[col].max()}")


# ### Updating the datatype of the datetime columns

# In[16]:


datetime_columns = ['trip_creation_time', 'od_start_time', 'od_end_time']
for col in datetime_columns:
    df[col] = pd.to_datetime(df[col], unit='ns')


# In[17]:


df['trip_creation_time'].min(), df['od_end_time'].max()


# ## <span style='color: blue;'>3. Data Cleaning</span> 

# ### Handling Missing Values in the data-set

# ### Checking for null values 

# In[18]:


np.any(df.isnull())


# In[19]:


df.isnull().sum()


# In[20]:


missing_source_name = df.loc[df['source_name'].isnull(), 'source_center'].unique()
missing_source_name


# In[21]:


for i in missing_source_name:
    unique_source_name = df.loc[df['source_center'] == i, 'source_name'].unique()
    if pd.isna(unique_source_name):
        print("Source Center :", i, "-" * 10, "Source Name :", 'Not Found')
    else : 
        print("Source Center :", i, "-" * 10, "Source Name :", unique_source_name)


# In[22]:


missing_destination_name = df.loc[df['destination_name'].isnull(), 'destination_center'].unique()
missing_destination_name


# In[23]:


for i in missing_destination_name:
    unique_destination_name = df.loc[df['destination_center'] == i, 'destination_name'].unique()
    if pd.isna(unique_source_name):
        print("destination_center :", i, "-" * 10, "destination_name :", 'Not Found')
    else : 
        print("destination_center :", i, "-" * 10, "destination_name :", unique_destination_name)


# ### The IDs for which the source name is missing, are all those IDs for destination also missing ?

# In[24]:


np.all(df.loc[df['source_name'].isnull(), 'source_center'].isin(missing_destination_name))


# ### Treating missing destination names and source names
# 

# In[25]:


count = 1
for i in missing_destination_name:
    df.loc[df['destination_center'] == i, 'destination_name'] = df.loc[df['destination_center'] == i, 
   'destination_name'].replace(np.nan, f'location_{count}')
    count += 1


# In[26]:


d = {}
for i in missing_source_name:
    d[i] = df.loc[df['destination_center'] == i, 'destination_name'].unique()
for idx, val in d.items():
    if len(val) == 0:
        d[idx] = [f'location_{count}']
        count += 1
d2 = {}
for idx, val in d.items():
    d2[idx] = val[0]
for i, v in d2.items():
    print(i, v)


# In[27]:


for i in missing_source_name:
    df.loc[df['source_center'] == i, 'source_name'] = df.loc[df['source_center'] == i, 'source_name'].replace(np.nan, d2[i])


# In[28]:


df.isna().sum()


# # Statistical Summary

# In[29]:


df.describe()


# In[30]:


df.describe(include = 'object')


# ## <span style='color: blue;'>4. Building features to prepare the data for actual analysis</span> 

# ### 1. grouping Trip_uuid,'source_center', and'destination_center'

# In[31]:


combined_package_details = ['trip_uuid', 'source_center', 'destination_center']
#Creating another dataframe to merge the above-mentioned features
df1 = df.groupby(by = combined_package_details, as_index = False).agg({'data' : 'first',
                                                         'route_type' : 'first',
                                                       'trip_creation_time' : 'first',
                                                       'source_name' : 'first',
                                                       'destination_name' : 'last',
                                                       'od_start_time' : 'first',
                                                       'od_end_time' : 'first',
                                                       'start_scan_to_end_scan' : 'first',
                                                       'actual_distance_to_destination' : 'last',
                                                       'actual_time' : 'last',
                                                       'osrm_time' : 'last',
                                                       'osrm_distance' : 'last',
                                                       'segment_actual_time' : 'sum',
                                                       'segment_osrm_time' : 'sum',
                                                       'segment_osrm_distance' : 'sum'})


# In[32]:


df1.head()


# ##  <span style='color: blue;'> Calculating the time taken between od_start_time and od_end_time and creating it as a feature. Also, dropping the original columns </span> 

# In[33]:


#New Feature name is: od_total_time
df1['od_total_time'] = df1['od_end_time'] - df1['od_start_time']
df1.drop(columns = ['od_end_time', 'od_start_time'], inplace = True)
df1['od_total_time'] = df1['od_total_time'].apply(lambda x: round(x.total_seconds()/60.0,2))
df1['od_total_time'].head()


# ## Further aggregate on the basis of just Trip_uuid

# In[34]:


df2 = df1.groupby(by = 'trip_uuid', as_index = False).agg({'source_center' : 'first',
                                                           'destination_center' : 'last',
                                                           'data' : 'first',
                                                           'route_type' : 'first',
                                                           'trip_creation_time' : 'first',
                                                           'source_name' : 'first',
                                                           'destination_name' : 'last',
                                                           'od_total_time' : 'sum',
                                                           'start_scan_to_end_scan' : 'sum',
                                                           'actual_distance_to_destination' : 'sum',
                                                           'actual_time' : 'sum',
                                                           'osrm_time' : 'sum',
                                                           'osrm_distance' : 'sum',
                                                           'segment_actual_time' : 'sum',
                                                           'segment_osrm_time' : 'sum',
                                                           'segment_osrm_distance' : 'sum'})
df2


# ##  <span style='color: blue;'> 5. Extracting Features</span> 

# ### 1. Source Name: Split and extract features out of 'source_name' - State | City | Place | Code 

# ### Extracting State

# In[35]:


def location_name_to_state(x):
    l = x.split('(')
    if len(l) == 1:
        return l[0]
    else:
        return l[1].replace(')', "")


# In[36]:


df2['source_state'] = df2['source_name'].apply(location_name_to_state)
df2['source_state'].unique()


# ### Extracting City

# In[38]:


def location_name_to_city(x):
    if 'location' in x:
        return 'unknown_city'
    else:
        l = x.split()[0].split('_')
        if 'CCU' in x:
            return 'Kolkata'
        elif 'MAA' in x.upper():
            return 'Chennai'
        elif ('HBR' in x.upper()) or ('BLR' in x.upper()):
            return 'Bengaluru'
        elif 'FBD' in x.upper():
            return 'Faridabad'
        elif 'BOM' in x.upper():
            return 'Mumbai'
        elif 'DEL' in x.upper():
            return 'Delhi'
        elif 'OK' in x.upper():
            return 'Delhi'
        elif 'GZB' in x.upper():
            return 'Ghaziabad'
        elif 'GGN' in x.upper():
            return 'Gurgaon'
        elif 'AMD' in x.upper():
            return 'Ahmedabad'
        elif 'CJB' in x.upper():
            return 'Coimbatore'
        elif 'HYD' in x.upper():
            return 'Hyderabad'
        return l[0]


# In[39]:


df2['source_city'] = df2['source_name'].apply(location_name_to_city)
print('No of source cities :', df2['source_city'].nunique())
df2['source_city'].unique()[:100]


# ### Extracting Place

# In[40]:


def location_name_to_place(x):
    '''if 'location' in x:
        return x
    elif 'HBR' in x:
        return 'HBR Layout PC'
    else:
        l = x.split()[0].split('_', 1)
        if len(l) == 1:
            return 'unknown_place'
        else:
            return l[1]
         '''
        # we will remove state
    x = x.split('(')[0]

    len_ = len(x.split('_'))

    if len_ >= 3:
        return x.split('_')[1]

    # small cities have same city and place name
    if len_ == 2:
        return x.split('_')[0]

    # now we need to deal with edge cases or imporper name convention

    # if len(x.split('_')) == 2:

    return x.split(' ')[0]


# In[41]:


df2['source_place'] = df2['source_name'].apply(location_name_to_place)
df2['source_place'].unique()[:100]


# ### Extracting Code

# In[42]:


def location_name_to_code(x):
    # we will remove state
    x = x.split('(')[0]

    if len(x.split('_')) >= 3:
        return x.split('_')[-1]

    return 'none'


# In[43]:


df2['source_code'] = df2['source_name'].apply(location_name_to_code)
df2['source_code'].unique()[:100]


# ### Source: State - City - Place - Code Table

# In[44]:


df2[['source_state','source_city','source_place','source_code']].head(10)


# ### 2. Destination Name: Split and extract features out of 'destination_name' - State | City | Place | Code

# In[45]:


df2['destination_state'] = df2['destination_name'].apply(location_name_to_state)
df2['destination_state'].head(10)


# In[46]:


df2['destination_city'] = df2['destination_name'].apply(location_name_to_city)
df2['destination_city'].head()


# In[47]:


df2['destination_place'] = df2['destination_name'].apply(location_name_to_place)
df2['destination_place'].head()


# In[48]:


df2['destination_code'] = df2['destination_name'].apply(location_name_to_code)
df2['destination_code'].head()


# ### Destination: State - City - Place - Code Table

# In[49]:


df2[['destination_state','destination_city','destination_place','destination_code']].head(10)


# ### 3. Extracting features Hour | date | day | week | month | year  from 'Trip_creation_time'

# In[50]:


df2['trip_creation_hour'] = df2['trip_creation_time'].dt.hour
df2['trip_creation_hour'] = df2['trip_creation_hour'].astype('int8')
df2['trip_creation_hour'].head()


# In[51]:


df2['trip_creation_date'] = pd.to_datetime(df2['trip_creation_time'].dt.date)
df2['trip_creation_date'].head()


# In[52]:


df2['trip_creation_day'] = df2['trip_creation_time'].dt.day
df2['trip_creation_day'] = df2['trip_creation_day'].astype('int8')
df2['trip_creation_day'].head()


# In[53]:


df2['trip_creation_week'] = df2['trip_creation_time'].dt.isocalendar().week
df2['trip_creation_week'] = df2['trip_creation_week'].astype('int8')
df2['trip_creation_week'].head()


# In[54]:


df2['trip_creation_month'] = df2['trip_creation_time'].dt.month
df2['trip_creation_month'] = df2['trip_creation_month'].astype('int8')
df2['trip_creation_month'].head()


# In[55]:


df2['trip_creation_year'] = df2['trip_creation_time'].dt.year
df2['trip_creation_year'] = df2['trip_creation_year'].astype('int16')
df2['trip_creation_year'].head()


# # Finding the structure of data after data cleaning

# In[56]:


df2.shape


# In[57]:


df2.info()


# In[58]:


df2.describe().T


# In[59]:


df2.describe(include = object).T


# ## <span style='color: blue;'> 6. Data Visualization | Data Analysis</span>  

# ## <span style='color: blue;'>Q1. How many trips are created on the hourly basis?</span> 
# 

# In[60]:


df2['trip_creation_hour'].unique()


# In[61]:


df_hour = df2.groupby(by = 'trip_creation_hour')['trip_uuid'].count().to_frame().reset_index()
df_hour.head()


# In[62]:


plt.figure(figsize = (12,6))
sns.lineplot(data = df_hour, 
            x = df_hour['trip_creation_hour'],
            y = df_hour['trip_uuid'],
            markers = '*')
plt.xticks(np.arange(0,24))
plt.grid('both')
plt.plot()


# ### Insights: 
# #### 1. It can be inferred from the above plot that the number of trips start increasing post 12 pm.
# #### 2. Number of trips observed to be maximum at 10 P.M and then start decreasing

# ## <span style='color: blue;'>Q2.How many trips are created for different days of the month?</span> 

# In[63]:


df2['trip_creation_day'].unique()


# In[64]:


df_day = df2.groupby(by = 'trip_creation_day')['trip_uuid'].count().to_frame().reset_index()
df_day.head()


# In[65]:


plt.figure(figsize = (15, 6))
sns.lineplot(data = df_day, 
             x = df_day['trip_creation_day'], 
             y = df_day['trip_uuid'], 
             markers = 'o')
plt.xticks(np.arange(1, 32))
plt.grid('both')
plt.plot()


# ### Insights:
# #### 1. It can be observed from the above plot that most of the trips are created in the mid of the month.
# #### 2. That means customers usually request more orders during the mid of the month.

# ## <span style='color: blue;'>Q3. How many trips are created for different weeks?</span>  

# In[66]:


df2['trip_creation_week'].unique()


# In[67]:


df_week = df2.groupby(by = 'trip_creation_week')['trip_uuid'].count().to_frame().reset_index()
df_week.head()


# In[68]:


plt.figure(figsize = (12, 6))
sns.lineplot(data = df_week, 
             x = df_week['trip_creation_week'], 
             y = df_week['trip_uuid'], 
             markers = 'o')
plt.grid('both')
plt.plot()


# ### Insights: It can be inferred from the above plot that most of the trips are created in the 38th week.

# ## <span style='color: blue;'>Q4. How many trips are created in the given two months?</span>  

# In[69]:


df_month = df2.groupby(by = 'trip_creation_month')['trip_uuid'].count().to_frame().reset_index()
df_month['perc'] = np.round(df_month['trip_uuid'] * 100/ df_month['trip_uuid'].sum(), 2)
df_month.head()


# In[70]:


df_month = df2.groupby(by = 'trip_creation_month')['trip_uuid'].count().to_frame().reset_index()
df_month['percentage'] = np.round(df_month['trip_uuid'] * 100/ df_month['trip_uuid'].sum(), 2)
df_month.head()


# In[71]:


plt.pie(x = df_month['trip_uuid'], 
        labels = ['Sep', 'Oct'],
        explode = [0, 0.1],
       autopct = '%.2f%%')
plt.plot()


# #### Observations: Trips are created a way more than in September when compared to October

# ##  <span style='color: blue;'>Q5. What is the distribution of trip data for the orders?</span>  

# In[72]:


df_data = df2.groupby(by = 'data')['trip_uuid'].count().to_frame().reset_index()
df_data['perc'] = np.round(df_data['trip_uuid'] * 100/ df_data['trip_uuid'].sum(), 2)
df_data.head()


# In[73]:


plt.pie(x = df_data['trip_uuid'], 
        labels = df_data['data'],
        explode = [0, 0.1],
        autopct = '%.2f%%')
plt.plot()


# #### Observations: The dataset predominantly comprises the training data.

# ##  <span style='color: blue;'>Q6. What is the distribution of route types for the orders?</span>  

# In[74]:


df_route = df2.groupby(by = 'route_type')['trip_uuid'].count().to_frame().reset_index()
df_route['percentage'] = np.round(df_route['trip_uuid'] * 100/ df_route['trip_uuid'].sum(), 2)
df_route.head()


# In[75]:


plt.pie(x = df_route['trip_uuid'], 
        labels = ['Carting', 'FTL'],
        explode = [0, 0.1],
        autopct = '%.2f%%')
plt.plot()


# ### Insights:
# #### The distribution of route types reveals a significant prevalence of carting, constituting 60.12% of the total routes, while FTL (Full Truck Load) accounts for the remaining 39.88%. This insight suggests a higher frequency or preference for carting in comparison to FTL, possibly indicating specific operational or logistical considerations within the context of transportation or shipping activities. 

# ## <span style='color: blue;'>Q7. what is the distribution of number of trips created from different states?</span>   

# In[76]:


df_source_state = df2.groupby(by = 'source_state')['trip_uuid'].count().to_frame().reset_index()
df_source_state['perc'] = np.round(df_source_state['trip_uuid'] * 100/ df_source_state['trip_uuid'].sum(), 2)
df_source_state = df_source_state.sort_values(by = 'trip_uuid', ascending = False)
df_source_state.head()


# In[95]:


plt.figure(figsize=(8, 5))

# Select the top 10 rows
df_top10 = df_source_state.head(10)

# Create a bar plot for the top 10
ax = sns.barplot(data=df_top10,
                 x='trip_uuid',
                 y='source_state',
                 palette='viridis')

# Add labels to each bar
for index, value in enumerate(df_top10['trip_uuid']):
    ax.text(value, index, f'{value} ({df_top10["perc"].iloc[index]}%)', va='center')

plt.title('Trips created from different States (Top 10)')
plt.xlabel('Number of Trips')
plt.ylabel('Source State')
plt.show()


# ## <span style='color: blue;'>Q8. Find the top 10 cities based on the number of trips created from different cities?</span>   

# In[87]:


df_source_city = df2.groupby(by = 'source_city')['trip_uuid'].count().to_frame().reset_index()
df_source_city['perc'] = np.round(df_source_city['trip_uuid'] * 100/ df_source_city['trip_uuid'].sum(), 2)
df_source_city = df_source_city.sort_values(by = 'trip_uuid', ascending = False)[:10]
df_source_city


# In[90]:


plt.figure(figsize=(8, 5))

# Create a bar plot
ax = sns.barplot(data=df_source_city,
                 x='trip_uuid',
                 y='source_city',
                 palette='magma')  

# Add labels to each bar
for index, value in enumerate(df_source_city['trip_uuid']):
    ax.text(value, index, f'{value} ({df_source_city["perc"].iloc[index]}%)', va='center')

plt.title('Top 10 cities based on the number of trips')
plt.xlabel('Number of Trips')
plt.ylabel('Source City')
plt.show()


# ### Insights: The plotted data illustrates that the highest number of trips originated from Mumbai, followed by Gurgaon, Delhi, Bengaluru, and Bhiwandi. This pattern indicates a robust seller base in these cities, suggesting a significant presence and potential market strength in these geographical areas.

# ## <span style='color: blue;'>Q9. what is the distribution of number of trips which ended in different states?</span>   

# In[98]:


df_destination_state = df2.groupby(by = 'destination_state')['trip_uuid'].count().to_frame().reset_index()
df_destination_state['perc'] = np.round(df_destination_state['trip_uuid'] * 100/ df_destination_state['trip_uuid'].sum(), 2)
df_destination_state = df_destination_state.sort_values(by = 'trip_uuid', ascending = False)
df_destination_state.head()


# In[101]:


plt.figure(figsize=(8, 5))

df_top10 = df_destination_state.head(10)

# Create a bar plot
ax = sns.barplot(data=df_top10,
                 x='trip_uuid',
                 y='destination_state',
                 palette='twilight')  

# Add labels to each bar
for index, value in enumerate(df_top10['trip_uuid']):
    ax.text(value, index, f'{value} ({df_top10["perc"].iloc[index]}%)', va='center')

plt.title('Number of trips which ended in different states (top 10)')
plt.xlabel('Number of Trips')
plt.ylabel('Destination State')
plt.show()


# ### Insights:
# #### The plotted data reveals that the highest number of trips concluded in the Maharashtra state, followed by Karnataka, Haryana, Tamil Nadu, and Uttar Pradesh. This observation indicates a notably elevated frequency of orders in these states, suggesting a substantial demand or transaction volume in these regions.

# ## <span style='color: blue;'>Q10. Find the top 10 cities based on the number of trips ended in different cities?</span>   

# In[91]:


df_destination_city = df2.groupby(by = 'destination_city')['trip_uuid'].count().to_frame().reset_index()
df_destination_city['perc'] = np.round(df_destination_city['trip_uuid'] * 100/ df_destination_city['trip_uuid'].sum(), 2)
df_destination_city = df_destination_city.sort_values(by = 'trip_uuid', ascending = False)[:10]
df_destination_city


# In[93]:


plt.figure(figsize=(8, 5))

# Create a bar plot
ax = sns.barplot(data=df_destination_city,
                 x='trip_uuid',
                 y='destination_city',
                 palette='inferno')  

# Add labels to each bar
for index, value in enumerate(df_destination_city['trip_uuid']):
    ax.text(value, index, f'{value} ({df_destination_city["perc"].iloc[index]}%)', va='center')

plt.title('Top 10 cities based on the number of trips ended in different cities')
plt.xlabel('Number of Trips')
plt.ylabel('Destination City')
plt.show()


# ### Insights:
# #### The plotted data indicates that the majority of trips concluded in Mumbai city, with Bengaluru, Gurgaon, Delhi, and Chennai following closely. This pattern suggests a notably high volume of orders in these cities, highlighting a substantial level of demand or transaction activity.

# ##  <span style='color: blue;'>7. Distribution of the variables and relationship between them</span>

# In[84]:


numerical_columns = ['od_total_time', 'start_scan_to_end_scan', 'actual_distance_to_destination',
                    'actual_time', 'osrm_time', 'osrm_distance', 'segment_actual_time',
                    'segment_osrm_time', 'segment_osrm_distance']
sns.pairplot(data = df2,
             vars = numerical_columns,
             kind = 'reg',
             hue = 'route_type',
             markers = '.')
plt.plot()


# In[85]:


df_corr = df2[numerical_columns].corr()
df_corr


# In[86]:


plt.figure(figsize = (15, 10))
sns.heatmap(data = df_corr, vmin = -1, vmax = 1, annot = True)
plt.plot()


# ### Insights: Very High Correlation (> 0.9) exists between columns all the numerical columns specified above

# ## <span style='color: blue;'> 8. In-depth analysis and feature engineering</span>

# ## <span style='color: blue;'>a.) Comparing the difference between od_total_time and start_scan_to_end_scan. Do hypothesis testing/ Visual analysis to check</span>

# ### STEP-1: Set up Null Hypothesis

# #### Null Hypothesis ( H0 ) - od_total_time (Total Trip Time) and start_scan_to_end_scan (Expected total trip time) are same.
# 
# #### Alternate Hypothesis ( HA ) - od_total_time (Total Trip Time) and start_scan_to_end_scan (Expected total trip time) are different.

# ### STEP-2: Checking for basic assumpitons for the hypothesis

# ### Visual Tests to know if the samples follow normal distribution

# In[109]:


plt.figure(figsize = (8, 5))
sns.histplot(df2['od_total_time'], element = 'step', color = 'green')
sns.histplot(df2['start_scan_to_end_scan'], element = 'step', color = 'pink')
plt.legend(['od_total_time', 'start_scan_to_end_scan'])
plt.plot()


# #### Observation: The above plot does not follow the normal distribution

# ### Distribution check using QQ Plot

# In[110]:


plt.figure(figsize = (15, 6))
plt.subplot(1, 2, 1)
plt.suptitle('QQ plots for od_total_time and start_scan_to_end_scan')
spy.probplot(df2['od_total_time'], plot = plt, dist = 'norm')
plt.title('QQ plot for od_total_time')
plt.subplot(1, 2, 2)
spy.probplot(df2['start_scan_to_end_scan'], plot = plt, dist = 'norm')
plt.title('QQ plot for start_scan_to_end_scan')
plt.plot()


# #### Observation: The above plot that the samples do not come from normal distribution

# In[102]:


df2[['od_total_time', 'start_scan_to_end_scan']].describe()


# ### Applying Shapiro-Wilk test for normality

# #### H0 : The sample follows normal distribution 
# #### Ha : The sample does not follow normal distribution
# #### alpha = 0.05

# In[112]:


test_stat, p_value = spy.shapiro(df2['od_total_time'].sample(5000))
print('p-value', p_value)
if p_value < 0.05:
    print('The sample does not follow normal distribution')
else:
    print('The sample follows normal distribution')   


# In[113]:


test_stat, p_value = spy.shapiro(df2['start_scan_to_end_scan'].sample(5000))
print('p-value', p_value)
if p_value < 0.05:
    print('The sample does not follow normal distribution')
else:
    print('The sample follows normal distribution') 


# ### Homogeneity of Variances using Lavene's test

# #### Null Hypothesis(H0) - Homogenous Variance 
# 
# #### Alternate Hypothesis(HA) - Non Homogenous Variance 

# In[114]:


test_stat, p_value = spy.levene(df2['od_total_time'], df2['start_scan_to_end_scan'])
print('p-value', p_value)
if p_value < 0.05:
    print('The samples do not have  Homogenous Variance')
else:
    print('The samples have Homogenous Variance ') 


# ### Since the samples are not normally distributed, T-Test cannot be applied here
# ### we can perform its non parametric equivalent test i.e., Mann-Whitney U rank test for two independent samples.

# In[115]:


test_stat, p_value = spy.mannwhitneyu(df2['od_total_time'], df2['start_scan_to_end_scan'])
print('P-value :',p_value)
if p_value < 0.05:
    print('od_total_time and start_scan_to_end_scan are different')
else:
    print('od_total_time and start_scan_to_end_scan are similar') 


# ### Conclusion: od_total_time and start_scan_to_end_scan are similar

# ## <span style='color: blue;'>b.) Hypothesis testing / visual analysis between actual_time aggregated value and OSRM time aggregated value (aggregated values are the values after merging the rows on the basis of trip_uuid)</span>

# In[116]:


df2[['actual_time', 'osrm_time']].describe()


# ### Visual Tests to know if the samples follow normal distribution

# In[121]:


plt.figure(figsize = (8, 6))
sns.histplot(df2['actual_time'], element = 'step', color = 'green')
sns.histplot(df2['osrm_time'], element = 'step', color = 'lightblue')
plt.legend(['actual_time', 'osrm_time'])
plt.plot()


# #### Observation: The above plot does not follow the normal distribution

# ### Distribution check using QQ Plot

# In[122]:


plt.figure(figsize = (15, 6))
plt.subplot(1, 2, 1)
plt.suptitle('QQ plots for actual_time and osrm_time')
spy.probplot(df2['actual_time'], plot = plt, dist = 'norm')
plt.title('QQ plot for actual_time')
plt.subplot(1, 2, 2)
spy.probplot(df2['osrm_time'], plot = plt, dist = 'norm')
plt.title('QQ plot for osrm_time')
plt.plot()


# ### The above plots that the samples do not come from normal distribution.

# ### Applying Shapiro-Wilk test for normality

# #### H0 : The sample follows normal distribution 
# #### Ha : The sample does not follow normal distribution
# #### alpha = 0.05

# In[124]:


test_stat, p_value = spy.shapiro(df2['actual_time'].sample(5000))
print('p-value', p_value)
if p_value < 0.05:
    print('The sample does not follow normal distribution')
else:
    print('The sample follows normal distribution')  


# In[125]:


test_stat, p_value = spy.shapiro(df2['osrm_time'].sample(5000))
print('p-value', p_value)
if p_value < 0.05:
    print('The sample does not follow normal distribution')
else:
    print('The sample follows normal distribution')  


# ### Homogeneity of Variances using Lavene's test

# #### Null Hypothesis(H0) - Homogenous Variance 
# #### Alternate Hypothesis(HA) - Non Homogenous Variance 

# In[126]:


test_stat, p_value = spy.levene(df2['actual_time'], df2['osrm_time'])
print('p-value', p_value)
if p_value < 0.05:
    print('The samples do not have  Homogenous Variance')
else:
    print('The samples have Homogenous Variance ') 


# #### Since the samples do not follow any of the assumptions T-Test cannot be applied here, we can perform its non parametric equivalent test i.e., Mann-Whitney U rank test for two independent samples.

# In[127]:


test_stat, p_value = spy.mannwhitneyu(df2['actual_time'], df2['osrm_time'])
print('p-value', p_value)
if p_value < 0.05:
    print('The samples are not similar')
else:
    print('The samples are similar ') 


# ### Conclusion: Since p-value < alpha therfore it can be concluded that actual_time and osrm_time are not similar.

# ## <span style='color: blue;'> c.) Do hypothesis testing/ visual analysis between actual_time aggregated value and segment actual time aggregated value (aggregated values are the values you’ll get after merging the rows on the basis of trip_uuid)</span>

# In[128]:


df2[['actual_time', 'segment_actual_time']].describe()


# ### Visual Tests to know if the samples follow normal distribution

# In[130]:


plt.figure(figsize = (8, 5))
sns.histplot(df2['actual_time'], element = 'step', color = 'magenta')
sns.histplot(df2['segment_actual_time'], element = 'step', color = 'lightgreen')
plt.legend(['actual_time', 'segment_actual_time'])
plt.plot()


# #### Observation: The above plot does not follow the normal distribution

# ### Distribution check using QQ Plot

# In[131]:


plt.figure(figsize = (15, 6))
plt.subplot(1, 2, 1)
plt.suptitle('QQ plots for actual_time and segment_actual_time')
spy.probplot(df2['actual_time'], plot = plt, dist = 'norm')
plt.title('QQ plot for actual_time')
plt.subplot(1, 2, 2)
spy.probplot(df2['segment_actual_time'], plot = plt, dist = 'norm')
plt.title('QQ plot for segment_actual_time')
plt.plot()


# ### Observation: The above plot that the samples do not come from normal distribution.

# ### Applying Shapiro-Wilk test for normality

# #### H0 : The sample follows normal distribution 
# #### Ha : The sample does not follow normal distribution
# #### alpha = 0.05

# In[133]:


test_stat, p_value = spy.shapiro(df2['actual_time'].sample(5000))
print('p-value', p_value)
if p_value < 0.05:
    print('The sample does not follow normal distribution')
else:
    print('The sample follows normal distribution')  


# In[134]:


test_stat, p_value = spy.shapiro(df2['segment_actual_time'].sample(5000))
print('p-value', p_value)
if p_value < 0.05:
    print('The sample does not follow normal distribution')
else:
    print('The sample follows normal distribution')  


# ### Homogeneity of Variances using Lavene's test

# #### Null Hypothesis(H0) - Homogenous Variance 
# #### Alternate Hypothesis(HA) - Non Homogenous Variance 

# In[135]:


test_stat, p_value = spy.levene(df2['actual_time'], df2['segment_actual_time'])
print('p-value', p_value)

if p_value < 0.05:
    print('The samples do not have Homogenous Variance')
else:
    print('The samples have Homogenous Variance ') 


# ### Since the samples do not come from normal distribution T-Test cannot be applied here, we can perform its non parametric equivalent test i.e., Mann-Whitney U rank test for two independent samples.

# In[136]:


test_stat, p_value = spy.mannwhitneyu(df2['actual_time'], df2['segment_actual_time'])
print('p-value', p_value)
if p_value < 0.05:
    print('The samples are not similar')
else:
    print('The samples are similar ') 


# ### Conclusion: Since p-value > alpha therfore it can be concluded that actual_time and segment_actual_time are similar.

# ##  <span style='color: blue;'> d.) hypothesis testing/ visual analysis between osrm distance aggregated value and segment osrm distance aggregated value (aggregated values are the values after merging the rows on the basis of trip_uuid) </span> 

# In[138]:


df2[['osrm_distance', 'segment_osrm_distance']].describe()


# ### Visual Tests to know if the samples follow normal distribution

# In[140]:


plt.figure(figsize = (8, 5))
sns.histplot(df2['osrm_distance'], element = 'step', color = 'green', bins = 1000)
sns.histplot(df2['segment_osrm_distance'], element = 'step', color = 'pink', bins = 1000)
plt.legend(['osrm_distance', 'segment_osrm_distance'])
plt.plot()


# ### Observation: The above plot does not follow the normal distribution

# ### Distribution check using QQ Plot

# In[141]:


plt.figure(figsize = (15, 6))
plt.subplot(1, 2, 1)
plt.suptitle('QQ plots for osrm_distance and segment_osrm_distance')
spy.probplot(df2['osrm_distance'], plot = plt, dist = 'norm')
plt.title('QQ plot for osrm_distance')
plt.subplot(1, 2, 2)
spy.probplot(df2['segment_osrm_distance'], plot = plt, dist = 'norm')
plt.title('QQ plot for segment_osrm_distance')
plt.plot()


# #### Observation: The above plots that the samples do not come from normal distribution.

# ### Applying Shapiro-Wilk test for normality

# #### H0: The sample follows normal distribution 
# #### Ha: The sample does not follow normal distribution
# #### alpha = 0.05

# In[143]:


test_stat, p_value = spy.shapiro(df2['osrm_distance'].sample(5000))
print('p-value', p_value)
if p_value < 0.05:
    print('The sample does not follow normal distribution')
else:
    print('The sample follows normal distribution')    


# In[144]:


test_stat, p_value = spy.shapiro(df2['segment_osrm_distance'].sample(5000))
print('p-value', p_value)
if p_value < 0.05:
    print('The sample does not follow normal distribution')
else:
    print('The sample follows normal distribution') 


# ### Homogeneity of Variances using Lavene's test

# #### Null Hypothesis(H0) - Homogenous Variance 
# #### Alternate Hypothesis(HA) - Non Homogenous Variance

# In[146]:


test_stat, p_value = spy.levene(df2['osrm_distance'], df2['segment_osrm_distance'])
print('p-value', p_value)

if p_value < 0.05:
    print('The samples do not have Homogenous Variance')
else:
    print('The samples have Homogenous Variance ') 


# ### Since the samples do not follow any of the assumptions, T-Test cannot be applied here. We can perform its non parametric equivalent test i.e., Mann-Whitney U rank test for two independent samples.

# In[147]:


test_stat, p_value = spy.mannwhitneyu(df2['osrm_distance'], df2['segment_osrm_distance'])
print('p-value', p_value)
if p_value < 0.05:
    print('The samples are not similar')
else:
    print('The samples are similar ') 


# ### Conclusion: Since p-value < alpha therfore it can be concluded that osrm_distance and segment_osrm_distance are not similar.

# ## <span style='color: blue;'> e.) Hypothesis testing/ visual analysis between osrm time aggregated value and segment osrm time aggregated value (aggregated values are the values after merging the rows on the basis of trip_uuid) </span> 

# In[148]:


df2[['osrm_time', 'segment_osrm_time']].describe().T


# ### Visual Tests to know if the samples follow normal distribution

# In[149]:


plt.figure(figsize = (8, 5))
sns.histplot(df2['osrm_time'], element = 'step', color = 'green', bins = 1000)
sns.histplot(df2['segment_osrm_time'], element = 'step', color = 'pink', bins = 1000)
plt.legend(['osrm_time', 'segment_osrm_time'])
plt.plot()


# #### Observation: The above plot does not follow the normal distribution

# ### Distribution check using QQ Plot

# In[150]:


plt.figure(figsize = (15, 6))
plt.subplot(1, 2, 1)
plt.suptitle('QQ plots for osrm_time and segment_osrm_time')
spy.probplot(df2['osrm_time'], plot = plt, dist = 'norm')
plt.title('QQ plot for osrm_time')
plt.subplot(1, 2, 2)
spy.probplot(df2['segment_osrm_time'], plot = plt, dist = 'norm')
plt.title('QQ plot for segment_osrm_time')
plt.plot()


# ### Observation: The above plots that the samples do not come from normal distribution.

# ### Applying Shapiro-Wilk test for normality

# #### H0: The sample follows normal distribution 
# #### Ha: The sample does not follow normal distribution
# #### alpha = 0.05

# In[151]:


test_stat, p_value = spy.shapiro(df2['osrm_time'].sample(5000))
print('p-value', p_value)
if p_value < 0.05:
    print('The sample does not follow normal distribution')
else:
    print('The sample follows normal distribution')  


# In[152]:


test_stat, p_value = spy.shapiro(df2['segment_osrm_time'].sample(5000))
print('p-value', p_value)
if p_value < 0.05:
    print('The sample does not follow normal distribution')
else:
    print('The sample follows normal distribution')  


# ### Homogeneity of Variances using Lavene's test

# #### Null Hypothesis(H0) - Homogenous Variance 
# #### Alternate Hypothesis(HA) - Non Homogenous Variance 

# In[153]:


test_stat, p_value = spy.levene(df2['osrm_time'], df2['segment_osrm_time'])
print('p-value', p_value)

if p_value < 0.05:
    print('The samples do not have Homogenous Variance')
else:
    print('The samples have Homogenous Variance ') 


# ### Since the samples do not follow any of the assumptions, T-Test cannot be applied here. We can perform its non parametric equivalent test i.e., Mann-Whitney U rank test for two independent samples.

# In[154]:


test_stat, p_value = spy.mannwhitneyu(df2['osrm_time'], df2['segment_osrm_time'])
print('p-value', p_value)
if p_value < 0.05:
    print('The samples are not similar')
else:
    print('The samples are similar ') 


# ### Conclusion: Since p-value < alpha therfore it can be concluded that osrm_time and segment_osrm_time are not similar.

# ##  <span style='color: blue;'> 8. Outlier Treatment </span> 

# ## Finding outliers in the numerical variables and checking it using visual analysis

# In[155]:


numerical_columns = ['od_total_time', 'start_scan_to_end_scan', 'actual_distance_to_destination',
                    'actual_time', 'osrm_time', 'osrm_distance', 'segment_actual_time',
                    'segment_osrm_time', 'segment_osrm_distance']
df2[numerical_columns].describe().T


# In[156]:


plt.figure(figsize = (18, 15))
for i in range(len(numerical_columns)):
    plt.subplot(3, 3, i + 1)
    clr = np.random.choice(list(mpl.colors.cnames))
    sns.histplot(df2[numerical_columns[i]], bins = 1000, kde = True, color = clr)
    plt.title(f"Distribution of {numerical_columns[i]} column")
    plt.plot()


# ### Observation: It can be inferred from the above plots that data in all the numerical columns are right skewed.

# In[157]:


plt.figure(figsize = (18, 15))
for i in range(len(numerical_columns)):
    plt.subplot(3, 3, i + 1)
    clr = np.random.choice(list(mpl.colors.cnames))
    sns.boxplot(df2[numerical_columns[i]], color = clr)
    plt.title(f"Distribution of {numerical_columns[i]} column")
    plt.plot()


# ### Observation: The above plots that there are outliers in all the numerical columns that need to be treated.

# ### Handling the outliers using the IQR method

# In[158]:


for i in numerical_columns:
    Q1 = np.quantile(df2[i], 0.25)
    Q3 = np.quantile(df2[i], 0.75)
    IQR = Q3 - Q1
    LB = Q1 - 1.5 * IQR
    UB = Q3 + 1.5 * IQR
    outliers = df2.loc[(df2[i] < LB) | (df2[i] > UB)]
    print('Column :', i)
    print(f'Q1 : {Q1}')
    print(f'Q3 : {Q3}')
    print(f'IQR : {IQR}')
    print(f'LB : {LB}')
    print(f'UB : {UB}')
    print(f'Number of outliers : {outliers.shape[0]}')
    print('----------------------------------')


# #### Conclusion: The outliers present in our sample data can be the true outliers. It's best to remove outliers only when there is a sound reason for doing so. Some outliers represent natural variations in the population, and they should be left as is in the dataset.

# ## <span style='color: blue;'> 8. Dealing with categorical data by one-hot encoding of categorical variables</span>

# In[159]:


# Get value counts before one-hot encoding

df2['route_type'].value_counts()


# In[160]:


# Perform one-hot encoding on categorical column route type
from sklearn.preprocessing import LabelEncoder 
label_encoder = LabelEncoder()
df2['route_type'] = label_encoder.fit_transform(df2['route_type'])


# In[161]:


# Get value counts after one-hot encoding

df2['route_type'].value_counts()


# In[162]:


# Get value counts of categorical variable 'data' before one-hot encoding

df2['data'].value_counts()


# In[163]:


# Performing one-hot encoding on categorical variable 'data'
label_encoder = LabelEncoder()
df2['data'] = label_encoder.fit_transform(df2['data'])


# In[164]:


# Get value counts after one-hot encoding

df2['data'].value_counts()


# ##  <span style='color: blue;'> 9. Normalize/ Standardize the numerical features using MinMaxScaler or StandardScaler</span>

# In[166]:


#importing required library
from sklearn.preprocessing import MinMaxScaler


# In[167]:


plt.figure(figsize = (10, 6))
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df2['od_total_time'].to_numpy().reshape(-1, 1))
sns.histplot(scaled)
plt.title(f"Normalized {df2['od_total_time']} column")
plt.legend('od_total_time')
plt.plot()


# In[168]:


plt.figure(figsize = (10, 6))
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df2['start_scan_to_end_scan'].to_numpy().reshape(-1, 1))
sns.histplot(scaled)
plt.title(f"Normalized {df2['start_scan_to_end_scan']} column")
plt.plot()


# In[169]:


plt.figure(figsize = (10, 6))
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df2['actual_distance_to_destination'].to_numpy().reshape(-1, 1))
sns.histplot(scaled)
plt.title(f"Normalized {df2['actual_distance_to_destination']} column")
plt.plot()


# In[170]:


plt.figure(figsize = (10, 6))
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df2['actual_time'].to_numpy().reshape(-1, 1))
sns.histplot(scaled)
plt.title(f"Normalized {df2['actual_time']} column")
plt.plot()


# In[171]:


plt.figure(figsize = (10, 6))
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df2['osrm_time'].to_numpy().reshape(-1, 1))
sns.histplot(scaled)
plt.title(f"Normalized {df2['osrm_time']} column")
plt.plot()


# In[172]:


plt.figure(figsize = (10, 6))
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df2['osrm_distance'].to_numpy().reshape(-1, 1))
sns.histplot(scaled)
plt.title(f"Normalized {df2['osrm_distance']} column")
plt.plot()


# In[173]:


plt.figure(figsize = (10, 6))
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df2['segment_actual_time'].to_numpy().reshape(-1, 1))
sns.histplot(scaled)
plt.title(f"Normalized {df2['segment_actual_time']} column")
plt.plot()


# In[174]:


plt.figure(figsize = (10, 6))
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df2['segment_osrm_time'].to_numpy().reshape(-1, 1))
sns.histplot(scaled)
plt.title(f"Normalized {df2['segment_osrm_time']} column")
plt.plot()


# In[175]:


plt.figure(figsize = (10, 6))
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df2['segment_osrm_distance'].to_numpy().reshape(-1, 1))
sns.histplot(scaled)
plt.title(f"Normalized {df2['segment_osrm_distance']} column")
plt.plot()


# ## Column Standardization

# In[176]:


from sklearn.preprocessing import StandardScaler


# In[177]:


plt.figure(figsize = (10, 6))
# define standard scaler
scaler = StandardScaler()
# transform data
scaled = scaler.fit_transform(df2['od_total_time'].to_numpy().reshape(-1, 1))
sns.histplot(scaled)
plt.title(f"Standardized {df2['od_total_time']} column")
plt.legend('od_total_time')
plt.plot()


# In[178]:


plt.figure(figsize = (10, 6))
scaler = StandardScaler()
scaled = scaler.fit_transform(df2['start_scan_to_end_scan'].to_numpy().reshape(-1, 1))
sns.histplot(scaled)
plt.title(f"Standardized {df2['start_scan_to_end_scan']} column")
plt.plot()


# In[179]:


plt.figure(figsize = (10, 6))
scaler = StandardScaler()
scaled = scaler.fit_transform(df2['actual_distance_to_destination'].to_numpy().reshape(-1, 1))
sns.histplot(scaled)
plt.title(f"Standardized {df2['actual_distance_to_destination']} column")
plt.plot()


# In[180]:


plt.figure(figsize = (10, 6))
scaler = StandardScaler()
scaled = scaler.fit_transform(df2['actual_time'].to_numpy().reshape(-1, 1))
sns.histplot(scaled)
plt.title(f"Standardized {df2['actual_time']} column")
plt.plot()


# In[181]:


plt.figure(figsize = (10, 6))
scaler = StandardScaler()
scaled = scaler.fit_transform(df2['osrm_time'].to_numpy().reshape(-1, 1))
sns.histplot(scaled)
plt.title(f"Standardized {df2['osrm_time']} column")
plt.plot()


# In[182]:


plt.figure(figsize = (10, 6))
scaler = StandardScaler()
scaled = scaler.fit_transform(df2['osrm_distance'].to_numpy().reshape(-1, 1))
sns.histplot(scaled)
plt.title(f"Standardized {df2['osrm_distance']} column")
plt.plot()


# In[183]:


plt.figure(figsize = (10, 6))
scaler = StandardScaler()
scaled = scaler.fit_transform(df2['segment_actual_time'].to_numpy().reshape(-1, 1))
sns.histplot(scaled)
plt.title(f"Standardized {df2['segment_actual_time']} column")
plt.plot()


# In[184]:


plt.figure(figsize = (10, 6))
scaler = StandardScaler()
scaled = scaler.fit_transform(df2['segment_osrm_time'].to_numpy().reshape(-1, 1))
sns.histplot(scaled)
plt.title(f"Standardized {df2['segment_osrm_time']} column")
plt.plot()


# In[185]:


plt.figure(figsize = (10, 6))
scaler = StandardScaler()
scaled = scaler.fit_transform(df2['segment_osrm_distance'].to_numpy().reshape(-1, 1))
sns.histplot(scaled)
plt.title(f"Standardized {df2['segment_osrm_distance']} column")
plt.plot()


# ## <span style='color: blue;'> Business Insights </span>

# 1. The dataset spans from '2018-09-12 00:00:16' to '2018-10-08 03:00:24'.
# 
# 2. There are approximately 14,817 unique trip IDs, 1,508 unique source centers, 1,481 unique destination centers, 690 unique source cities, and 806 unique destination cities.
# 
# 3. Notably, the data is more oriented towards testing than training.
# 
# 4. The prevalent route type is Carting.
# 
# 5. Fourteen unique location IDs are absent in the dataset.
# 
# 6. Trip counts show an upward trend post-noon, peak at 10 P.M., and then decline.
# 
# 7. The highest number of trips occurred in the 38th week.
# 
# 8. A concentration of orders tends to occur in the middle of the month.
# 
# 9. Primary orders originate from states such as Maharashtra, Karnataka, Haryana, Tamil Nadu, and Telangana.
# 
# 10. The maximum number of trips originates from Mumbai, followed by Gurgaon Delhi, Bengaluru, and Bhiwandi, indicating a strong seller presence in these cities.
# 
# 11. Most trips conclude in Maharashtra, followed by Karnataka, Haryana, Tamil Nadu, and Uttar Pradesh, pointing to a significantly high order volume in these states.
# 
# 12. The majority of trips conclude in Mumbai, followed by Bengaluru, Gurgaon, Delhi, and Chennai, indicating substantial order activity in these cities.
# 
# 13. Regarding destination cities, the highest order frequency is observed in cities like Bengaluru, Mumbai, Gurgaon, Bangalore, and Delhi.
# 
# 14. The features 'start_scan_to_end_scan' and 'od_total_time' (created feature) exhibit statistical similarity.
# 
# 15. Conversely, features 'actual_time' and 'osrm_time' are statistically different.
# 
# 16. Features start_scan_to_end_scan and segment_actual_time are statistically similar.
# 
# 17. On the other hand, 'osrm_distance' and 'segment_osrm_distance' show statistical differences.
# 
# 18. Both the osrm_time & segment_osrm_time are not statistically same.

# ##  <span style='color: blue;'> Recommendations: </span>

# 1. There is a need for enhancements in the OSRM trip planning system to address discrepancies that may affect transporters when the routing engine is configured for optimal outcomes.
# 
# 2. Observing a variance between 'osrm_time' and 'actual_time,' it is essential for the team to minimize this difference. This reduction is crucial for more accurate delivery time predictions, ensuring greater convenience for customers in anticipating delivery windows.
# 
# 3. Discrepancies between 'osrm_distance' and the actual distance covered raise concerns. Possible explanations include deviations from predefined routes by delivery personnel or inaccuracies in the OSRM device's prediction of routes based on factors like distance and traffic. Investigation and corrective actions are warranted.
# 
# 4. A significant portion of orders originates from or is delivered to states such as Maharashtra, Karnataka, Haryana, and Tamil Nadu. To enhance service in these regions, existing corridors can be optimized further.
# 
# 5. For a deeper understanding of why major orders come from states like Maharashtra, Karnataka, Haryana, Tamil Nadu, and Uttar Pradesh, customer profiling is recommended. This analysis aims to improve the overall buying and delivery experience for customers in these states.
# 
# 6. Considering state-specific factors such as heavy traffic and challenging terrain conditions, this information serves as a valuable indicator for planning and catering to demand, particularly during peak festival seasons.

# ## ***************************************** End of Project ***********************************************

# In[ ]:




