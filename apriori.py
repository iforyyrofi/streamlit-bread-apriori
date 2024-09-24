import streamlit as st
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import association_rules, apriori

# load dataset
df = pd.read_csv('bread-basket.csv')
df['date_time'] = pd.to_datetime(df['date_time'], format='%d-%m-%Y %H:%M')

df['month'] = df['date_time'].dt.month
df['day'] = df['date_time'].dt.weekday

df['month'].replace([i for i in range(1, 12 + 1)], ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'], inplace=True)
df['day'].replace([i for i in range(6 + 1)], ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], inplace=True)

st.title('Market Basket Analysis Menggunakan Algoritma Apriori')

# Assumes period_day and weekday_weekend columns exist
def get_data(period_day='', weekday_weekend='', month='', day=''):
    data = df.copy()
    filtered = data.loc[
        (data['period_day'].str.contains(period_day)) & 
        (data['weekday_weekend'].str.contains(weekday_weekend)) & 
        (data['month'].str.contains(month)) & 
        (data['day'].str.contains(day))
    ]
    return filtered if filtered.shape[0] else 'No Result!'

def user_input_features():
    item = st.selectbox("item", ['Bread', 'Scandinavian', 'Hot chocolate', 'Jam', 'Cookies', 'Muffin', 'Coffee', 'Pastry', 'Medialuna', 'Tea', 'Tartine', 'Basket', 'Mineral water', 'Farm House', 'Fudge', 'Juice', "Ella's Kitchen Pouches", 'Victorian Sponge', 'Frittata', 'Hearty & Seasonal', 'Soup', 'Pick and Mix Bowls', 'Smoothies', 'Cake', 'Mighty Protein', 'Chicken sand', 'Coke', 'My-5 Fruit Shoot', 'Focaccia', 'Sandwich', 'Alfajores', 'Eggs', 'Brownie', 'Dulce de Leche', 'Honey', 'The BART', 'Granola', 'Fairy Doors', 'Empanadas', 'Keeping It Local', 'Art Tray', 'Bowl Nic Pitt', 'Bread Pudding', 'Adjustment', 'Truffles', 'Chimichurri Oil', 'Bacon', 'Spread', 'Kids biscuit', 'Siblings', 'Caramel bites', 'Jammie Dodgers', 'Tiffin', 'Olum & polenta', 'Polenta', 'The Nomad', 'Hack the stack', 'Bakewell', 'Lemon and coconut', 'Toast', 'Scone', 'Crepes', 'Vegan mincepie', 'Bare Popcorn', 'Muesli', 'Crisps', 'Pintxos', 'Gingerbread syrup', 'Panatone', 'Brioche and salami', 'Afternoon with the baker', 'Salad', 'Chicken Stew', 'Spanish Brunch', 'Raspberry shortbread sandwich', 'Extra Salami or Feta', 'Duck egg', 'Baguette', "Valentine's card", 'Tshirt', 'Vegan Feast', 'Postcard', 'Nomad bag', 'Chocolates', 'Coffee granules ', 'Drinking chocolate spoons ', 'Christmas common', 'Argentina Night', 'Half slice Monster ', 'Gift voucher', 'Cherry me Dried fruit', 'Mortimer', 'Raw bars', 'Tacos/Fajita'])
    period_day = st.selectbox('Period Day', ['Morning', 'Afternoon', 'Evening', 'Night'])
    weekday_weekend = st.selectbox('Weekday/Weekend', ['Weekend', 'Weekday'])
    month = st.select_slider('Month', ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])
    day = st.select_slider('Day', ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], value='Saturday')
    
    return item, period_day, weekday_weekend, month, day

item, period_day, weekday_weekend, month, day = user_input_features()

data = get_data(period_day.lower(), weekday_weekend.lower(), month, day)

def encode(x):
    return 1 if x >= 1 else 0

if type(data) != str:  # Check if data is not 'No Result!'
    item_count = data.groupby(['Transaction', 'Item'])['Item'].count().reset_index(name='Count')
    item_count_pivot = item_count.pivot_table(index='Transaction', columns='Item', values='Count', aggfunc='sum').fillna(0)
    item_count_pivot = item_count_pivot.applymap(encode)
    
    support = 0.01
    frequent_items = apriori(item_count_pivot, min_support=support, use_colnames=True)
    
    metric = 'lift'
    min_threshold = 1
    
    rules = association_rules(frequent_items, metric=metric, min_threshold=min_threshold)[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    rules.sort_values('confidence', ascending=False, inplace=True)

def parse_list(x):
    x = list(x)
    return ', '.join(x) if len(x) > 1 else x[0]

def return_item_df(item_antecedents):
    match_data = rules[rules['antecedents'].apply(parse_list) == item_antecedents]
    if match_data.shape[0] > 0:
        antecedent = parse_list(match_data.iloc[0]['antecedents'])
        consequent = parse_list(match_data.iloc[0]['consequents'])
        return antecedent, consequent
    return ["No Match", ""]


if type(data) != str:
    recommendation = return_item_df(item)
    st.markdown('Hasil Rekomendasi :')
    st.success(f'Jika konsumen membeli **{recommendation[0]}**, maka membeli **{recommendation[1]}** secara bersamaan.')




# cd
# D:
# cd "College\Semester 3\Fundamen Sains Data\Tugas\2\bread-apriori"
# streamlit run "apriori.py"
