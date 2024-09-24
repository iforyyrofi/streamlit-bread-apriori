import streamlit as st
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Load dataset
df = pd.read_csv('bread-basket.csv')
df['date_time'] = pd.to_datetime(df['date_time'], format='%d-%m-%Y %H:%M')

# Extract month and day
df['month'] = df['date_time'].dt.month
df['day'] = df['date_time'].dt.weekday

# Map month and day to string representations
month_mapping = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 
                 7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
df['month'] = df['month'].map(month_mapping)

day_mapping = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
df['day'] = df['day'].map(day_mapping)

st.title('Market Basket Analysis Using Apriori Algorithm')

def get_data(period_day='', weekday_weekend='', month='', day=''):
    data = df.copy()
    filtered = data.loc[
        (data['period_day'].str.contains(period_day)) &
        (data['weekday_weekend'].str.contains(weekday_weekend)) &
        (data['month'].str.contains(month)) &
        (data['day'].str.contains(day))
    ]
    return filtered if not filtered.empty else None

def user_input_features():
    item = st.selectbox("Item", [
        'Bread', 'Scandinavian', 'Hot chocolate', 'Jam', 'Cookies', 'Muffin', 
        'Coffee', 'Pastry', 'Medialuna', 'Tea', 'Tartine', 'Basket', 'Mineral water', 
        'Farm House', 'Fudge', 'Juice', "Ella's Kitchen Pouches", 'Victorian Sponge', 
        'Frittata', 'Hearty & Seasonal', 'Soup', 'Pick and Mix Bowls', 'Smoothies', 
        'Cake', 'Mighty Protein', 'Chicken sand', 'Coke', 'My-5 Fruit Shoot', 'Focaccia', 
        'Sandwich', 'Alfajores', 'Eggs', 'Brownie', 'Dulce de Leche', 'Honey', 
        'The BART', 'Granola', 'Fairy Doors', 'Empanadas', 'Keeping It Local', 
        'Art Tray', 'Bowl Nic Pitt', 'Bread Pudding', 'Adjustment', 'Truffles', 
        'Chimichurri Oil', 'Bacon', 'Spread', 'Kids biscuit', 'Siblings', 
        'Caramel bites', 'Jammie Dodgers', 'Tiffin', 'Olum & polenta', 'Polenta', 
        'The Nomad', 'Hack the stack', 'Bakewell', 'Lemon and coconut', 'Toast', 
        'Scone', 'Crepes', 'Vegan mincepie', 'Bare Popcorn', 'Muesli', 'Crisps', 
        'Pintxos', 'Gingerbread syrup', 'Panatone', 'Brioche and salami', 
        'Afternoon with the baker', 'Salad', 'Chicken Stew', 'Spanish Brunch', 
        'Raspberry shortbread sandwich', 'Extra Salami or Feta', 'Duck egg', 
        'Baguette', "Valentine's card", 'Tshirt', 'Vegan Feast', 'Postcard', 
        'Nomad bag', 'Chocolates', 'Coffee granules ', 'Drinking chocolate spoons ', 
        'Christmas common', 'Argentina Night', 'Half slice Monster ', 
        'Gift voucher', 'Cherry me Dried fruit', 'Mortimer', 'Raw bars', 'Tacos/Fajita'
    ])
    period_day = st.selectbox('Period Day', ['Morning', 'Afternoon', 'Evening', 'Night'])
    weekday_weekend = st.selectbox('Weekday/Weekend', ['Weekend', 'Weekday'])
    month = st.select_slider('Month', list(month_mapping.values()))
    day = st.select_slider('Day', list(day_mapping.values()), value='Saturday')

    return item, period_day, weekday_weekend, month, day

item, period_day, weekday_weekend, month, day = user_input_features()

data = get_data(period_day.lower(), weekday_weekend.lower(), month, day)

def encode(x):
    return 1 if x > 0 else 0

def generate_rules(data):
    if data is not None:
        item_count = data.groupby(['Transaction', 'Item'])['Item'].count().reset_index(name='Count')
        item_count_pivot = item_count.pivot_table(index='Transaction', columns='Item', values='Count', aggfunc='sum').fillna(0)
        item_count_pivot = item_count_pivot.applymap(encode)

        support = 0.01
        frequent_items = apriori(item_count_pivot, min_support=support, use_colnames=True)

        metric = 'lift'
        min_threshold = 1
        rules = association_rules(frequent_items, metric=metric, min_threshold=min_threshold)
        return rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    return None

rules = generate_rules(data)

def parse_list(x):
    if isinstance(x, frozenset):
        return ', '.join(list(x))
    elif len(x) == 1:
        return x[0]
    elif len(x) > 1:
        return ', '.join(x)

def return_item_df(item_antecedents):
    if rules is not None and not rules.empty:
        data = rules[['antecedents', 'consequents']].copy()
        data['antecedents'] = data['antecedents'].apply(parse_list)
        data['consequents'] = data['consequents'].apply(parse_list)

        matched_row = data.loc[data['antecedents'] == item_antecedents]
        if not matched_row.empty:
            return list(matched_row.iloc[0, :])
    return ['No recommendations available']


# (Rest of your code above remains unchanged)

if data is not None:
    st.markdown('Hasil Rekomendasi:')
    recommendation = return_item_df(item)

    # Check the length of recommendation before accessing its elements
    if len(recommendation) > 1:
        st.success(f'Jika konsumen membeli **{item}**, maka membeli **{recommendation[1]}** secara bersamaan')
    else:
        st.warning('Tidak ada rekomendasi tersedia untuk item ini.')
else:
    st.warning('No data found for the selected filters.')


# cd
# D:
# cd "College\Semester 3\Fundamen Sains Data\Tugas\2\bread-apriori"
# streamlit run "apriori.py"

