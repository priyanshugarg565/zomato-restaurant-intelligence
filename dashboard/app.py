import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import pickle, os
import base64
from PIL import Image
import io

# ── Load & prep data ──────────────────────────────────────────────────────────
df = pd.read_csv('df_clean.csv')
# df.drop(columns=['url','address','phone','reviews_list','menu_item'], inplace=True)
# df.drop_duplicates(inplace=True)
# df['rate'] = pd.to_numeric(df['rate'].astype(str).str.replace('/5','').str.strip(), errors='coerce')
# df['approx_cost(for two people)'] = pd.to_numeric(
#     df['approx_cost(for two people)'].astype(str).str.replace(',','').str.strip(), errors='coerce')
# df.rename(columns={'approx_cost(for two people)':'cost',
#                    'listed_in(type)':'listing_type',
#                    'listed_in(city)':'listed_city'}, inplace=True)
# df['online_order'] = (df['online_order']=='Yes').astype(int)
# df['book_table']   = (df['book_table']=='Yes').astype(int)
# df.dropna(subset=['rate','cost','location','cuisines','rest_type'], inplace=True)
# df['high_performer'] = ((df['rate']>=4.0) & (df['votes']>=200)).astype(int)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('le_dict.pkl', 'rb') as f:
    le_dict = pickle.load(f)

# # Encode for model
# df_enc = df.copy()
# le_dict = {}
# for col in ['location','rest_type','cuisines','listing_type','listed_city']:
#     le = LabelEncoder()
#     df_enc[col] = le.fit_transform(df_enc[col].astype(str))
#     le_dict[col] = le

# # Train model (quick retrain)
feature_cols = ['online_order','book_table','votes','cost',
                'location','rest_type','cuisines','listing_type']
# from sklearn.model_selection import train_test_split
# X = df_enc[feature_cols]; y = df_enc['high_performer']
# X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# scale_pos = (y==0).sum()/(y==1).sum()
# model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
#                       subsample=0.8, colsample_bytree=0.8,
#                       scale_pos_weight=scale_pos, random_state=42, eval_metric='logloss')
# model.fit(X_tr, y_tr, verbose=False)

# ── App layout ────────────────────────────────────────────────────────────────
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server

app.layout = dbc.Container(fluid=True, children=[

    dbc.Row(dbc.Col(html.H1("🍽️ Zomato Bangalore Restaurant Intelligence",
        className="text-center my-3",
        style={'color':'#FF6B6B','fontWeight':'bold','fontSize':'2rem'}))),

    # ── KPI Cards ──
    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardBody([
            html.H4("Total Restaurants", className="card-title text-muted"),
            html.H2(f"{len(df):,}", style={'color':'#FF6B6B'})])]), width=3),
        dbc.Col(dbc.Card([dbc.CardBody([
            html.H4("Avg Rating", className="card-title text-muted"),
            html.H2(f"{df['rate'].mean():.2f} ⭐", style={'color':'#4ECDC4'})])]), width=3),
        dbc.Col(dbc.Card([dbc.CardBody([
            html.H4("Online Order %", className="card-title text-muted"),
            html.H2(f"{df['online_order'].mean()*100:.1f}%", style={'color':'#45B7D1'})])]), width=3),
        dbc.Col(dbc.Card([dbc.CardBody([
            html.H4("High Performers", className="card-title text-muted"),
            html.H2(f"{df['high_performer'].sum():,}", style={'color':'#96CEB4'})])]), width=3),
    ], className="mb-4"),

    # ── Filters ──
    dbc.Row([
        dbc.Col([html.Label("Filter by Location", style={'color':'white'}),
            dcc.Dropdown(id='loc-filter',
                options=[{'label':'All','value':'All'}] +
                        [{'label':l,'value':l} for l in sorted(df['location'].unique())],
                value='All', clearable=False,
                style={'color':'black'})], width=4),
        dbc.Col([html.Label("Filter by Restaurant Type", style={'color':'white'}),
            dcc.Dropdown(id='type-filter',
                options=[{'label':'All','value':'All'}] +
                        [{'label':t,'value':t} for t in sorted(
                            df['rest_type'].str.split(',').explode().str.strip().unique())],
                value='All', clearable=False,
                style={'color':'black'})], width=4),
        dbc.Col([html.Label("Min Votes", style={'color':'white'}),
            dcc.Slider(id='votes-filter', min=0, max=500, step=50, value=0,
                marks={i:{'label':str(i),'style':{'color':'white'}} for i in range(0,501,100)})], width=4),
    ], className="mb-4"),

    # ── Charts Row 1 ──
    dbc.Row([
        dbc.Col(dcc.Graph(id='rating-dist'), width=6),
        dbc.Col(dcc.Graph(id='top-locations'), width=6),
    ], className="mb-3"),

    # ── Charts Row 2 ──
    dbc.Row([
        dbc.Col(dcc.Graph(id='cost-rating'), width=6),
        dbc.Col(dcc.Graph(id='cuisine-bar'), width=6),
    ], className="mb-3"),

    # ── SHAP Section ──
    dbc.Row(dbc.Col(html.H3("🧠 Model Explainability (SHAP)",
        style={'color':'#FF6B6B','fontWeight':'bold'}), className="mt-2 mb-3")),

    dbc.Row([
        dbc.Col([
            html.H5("Feature Importance", style={'color':'white','textAlign':'center'}),
            html.Img(src='/assets/shap_importance.png',
                style={'width':'100%','borderRadius':'8px'})
        ], width=6),
        dbc.Col([
            html.H5("SHAP Beeswarm — Impact Direction", style={'color':'white','textAlign':'center'}),
            html.Img(src='/assets/shap_beeswarm.png',
                style={'width':'100%','borderRadius':'8px'})
        ], width=6),
    ], className="mb-4"),

    dbc.Row(dbc.Col(dbc.Card(dbc.CardBody([
        html.H5("🔍 How to read this:", style={'color':'#FF6B6B'}),
        html.Ul([
            html.Li("votes is the dominant predictor — 10x more important than any other feature", style={'color':'white'}),
            html.Li("book_table = Yes strongly increases success probability", style={'color':'white'}),
            html.Li("online_order has near-zero impact despite 65.7% adoption", style={'color':'white'}),
            html.Li("High cost can go either way — both budget and premium restaurants can succeed", style={'color':'white'}),
        ])
    ]), style={'backgroundColor':'#2a2a2a'}), className="mb-4")),

    # ── Business Recommendations ──
    dbc.Row(dbc.Col(html.H3("💡 Business Recommendations Engine",
        style={'color':'#FF6B6B','fontWeight':'bold'}), className="mt-2 mb-3")),

    dbc.Row([
        dbc.Col([
            html.Label("I want to open a restaurant in:", style={'color':'white'}),
            dcc.Dropdown(id='rec-location',
                options=[{'label':l,'value':l} for l in sorted(df['location'].unique())],
                value='BTM', clearable=False, style={'color':'black'}),
        ], width=3),
        dbc.Col([
            html.Label("My budget for two (₹):", style={'color':'white'}),
            dcc.Slider(id='rec-budget', min=200, max=2000, step=100, value=800,
                marks={i:{'label':f'₹{i}','style':{'color':'white'}} for i in range(200,2001,400)}),
        ], width=4),
        dbc.Col([
            html.Label("Restaurant type:", style={'color':'white'}),
            dcc.Dropdown(id='rec-type',
                options=[{'label':'Any','value':'Any'}] +
                        [{'label':t,'value':t} for t in ['Quick Bites','Casual Dining','Cafe','Delivery','Dessert Parlor']],
                value='Any', clearable=False, style={'color':'black'}),
        ], width=3),
        dbc.Col([
            html.Br(),
            dbc.Button("Get Recommendations", id='rec-btn',
                color="danger", className="mt-2", style={'width':'100%'})
        ], width=2),
    ], className="mb-3"),

    dbc.Row(dbc.Col(html.Div(id='rec-output'), className="mb-5")),

    # ── Predictor ──
    dbc.Row(dbc.Col(html.H3("🔮 Restaurant Success Predictor",
        style={'color':'#FF6B6B','fontWeight':'bold'}), className="mt-2")),

    dbc.Row([
        dbc.Col([
            html.Label("Location", style={'color':'white'}),
            dcc.Dropdown(id='p-location',
                options=[{'label':l,'value':l} for l in sorted(df['location'].unique())],
                value='BTM', style={'color':'black'}),
            html.Label("Restaurant Type", style={'color':'white','marginTop':'10px'}),
            dcc.Dropdown(id='p-resttype',
                options=[{'label':t,'value':t} for t in sorted(
                    df['rest_type'].str.split(',').explode().str.strip().unique())],
                value='Casual Dining', style={'color':'black'}),
            html.Label("Cuisine", style={'color':'white','marginTop':'10px'}),
            dcc.Dropdown(id='p-cuisine',
                options=[{'label':c,'value':c} for c in sorted(
                    df['cuisines'].str.split(',').explode().str.strip().unique())],
                value='North Indian', style={'color':'black'}),
        ], width=3),
        dbc.Col([
            html.Label("Approx Cost for Two (₹)", style={'color':'white'}),
            dcc.Slider(id='p-cost', min=100, max=3000, step=100, value=500,
                marks={i:{'label':f'₹{i}','style':{'color':'white'}} 
                       for i in range(500,3001,500)}),
            html.Label("Expected Votes", style={'color':'white','marginTop':'20px'}),
            dcc.Slider(id='p-votes', min=0, max=2000, step=50, value=200,
                marks={i:{'label':str(i),'style':{'color':'white'}} 
                       for i in range(0,2001,400)}),
            html.Div([
                html.Label("Online Ordering", style={'color':'white','marginTop':'20px'}),
                dcc.RadioItems(id='p-online',
                    options=[{'label':' Yes','value':1},{'label':' No','value':0}],
                    value=1, inline=True,
                    style={'color':'white'},
                    labelStyle={'color':'white','marginRight':'15px'}),
                html.Label("Table Booking", style={'color':'white','marginTop':'10px'}),
                dcc.RadioItems(id='p-booktable',
                    options=[{'label':' Yes','value':1},{'label':' No','value':0}],
                    value=0, inline=True,
                    style={'color':'white'},
                    labelStyle={'color':'white','marginRight':'15px'}),
            ]),
        ], width=5),
        dbc.Col([
            html.Div(id='prediction-output', style={'marginTop':'10px'})
        ], width=4),
    ], className="mb-5"),

])

# ── Callbacks ─────────────────────────────────────────────────────────────────
def filter_df(location, rest_type, min_votes):
    dff = df.copy()
    if location != 'All': dff = dff[dff['location']==location]
    if rest_type != 'All': dff = dff[dff['rest_type'].str.contains(rest_type, na=False)]
    dff = dff[dff['votes'] >= min_votes]
    return dff

@app.callback(
    Output('rating-dist','figure'),
    Output('top-locations','figure'),
    Output('cost-rating','figure'),
    Output('cuisine-bar','figure'),
    Input('loc-filter','value'),
    Input('type-filter','value'),
    Input('votes-filter','value'))
def update_charts(location, rest_type, min_votes):
    dff = filter_df(location, rest_type, min_votes)

    # Rating distribution
    f1 = px.histogram(dff, x='rate', nbins=30, color_discrete_sequence=['#FF6B6B'],
        title='Rating Distribution', template='plotly_dark')
    f1.update_layout(bargap=0.1)

    # Top locations
    top_loc = dff['location'].value_counts().head(12).reset_index()
    top_loc.columns = ['location','count']
    f2 = px.bar(top_loc, x='count', y='location', orientation='h',
        color='count', color_continuous_scale='Reds',
        title='Top Locations by Restaurant Count', template='plotly_dark')
    f2.update_layout(yaxis={'categoryorder':'total ascending'})

    # Cost vs Rating scatter
    avg = dff.groupby('location').agg({'rate':'mean','cost':'mean','votes':'sum'}).reset_index()
    avg = avg[avg['votes']>500]
    f3 = px.scatter(avg, x='cost', y='rate', size='votes', color='rate',
        hover_name='location', color_continuous_scale='RdYlGn',
        title='Cost vs Rating by Location (Bubble = Votes)', template='plotly_dark',
        size_max=40)

    # Cuisine bar
    cuisine_counts = dff['cuisines'].str.split(',').explode().str.strip().value_counts().head(12).reset_index()
    cuisine_counts.columns = ['cuisine','count']
    f4 = px.bar(cuisine_counts, x='cuisine', y='count',
        color='count', color_continuous_scale='Teal',
        title='Top Cuisines', template='plotly_dark')
    f4.update_layout(xaxis_tickangle=45)

    return f1, f2, f3, f4

@app.callback(
    Output('prediction-output','children'),
    Input('p-location','value'),
    Input('p-resttype','value'),
    Input('p-cuisine','value'),
    Input('p-cost','value'),
    Input('p-votes','value'),
    Input('p-online','value'),
    Input('p-booktable','value'))
def predict(location, rest_type, cuisine, cost, votes, online, booktable):
    try:
        loc_enc = le_dict['location'].transform([location])[0]
        all_rest_types = le_dict['rest_type'].classes_
        rt_match = next((r for r in all_rest_types if rest_type in r), all_rest_types[0])
        rt_enc = le_dict['rest_type'].transform([rt_match])[0]
        # cu_enc  = le_dict['cuisines'].transform([cuisine])[0]
        # Find closest match in training data
        all_cuisines = le_dict['cuisines'].classes_
        match = next((c for c in all_cuisines if cuisine in c), all_cuisines[0])
        cu_enc = le_dict['cuisines'].transform([match])[0]
        lt_enc  = le_dict['listing_type'].transform(['Dine-out'])[0]

        X_input = pd.DataFrame([[online, booktable, votes, cost,
                                  loc_enc, rt_enc, cu_enc, lt_enc]],
                               columns=feature_cols)
        prob = model.predict_proba(X_input)[0][1]
        label = "🟢 High Performer" if prob >= 0.5 else "🔴 Unlikely High Performer"
        color = "#96CEB4" if prob >= 0.5 else "#FF6B6B"

        return dbc.Card(dbc.CardBody([
            html.H4("Prediction Result", className="text-muted"),
            html.H2(label, style={'color': color}),
            html.H3(f"Success Probability: {prob*100:.1f}%", style={'color': color}),
            dbc.Progress(value=prob*100, color="success" if prob>=0.5 else "danger",
                        style={'height':'20px','marginTop':'10px'}),
        ]), style={'backgroundColor':'#2a2a2a'})
    except Exception as e:
        return html.P(f"Select valid options to predict. ({e})", style={'color':'gray'})

@app.callback(
    Output('rec-output','children'),
    Input('rec-btn','n_clicks'),
    Input('rec-location','value'),
    Input('rec-budget','value'),
    Input('rec-type','value'),
    prevent_initial_call=False)
def get_recommendations(n_clicks, location, budget, rest_type):
    dff = df[df['location']==location].copy()
    dff = dff[dff['cost'] <= budget]
    if rest_type != 'Any':
        dff = dff[dff['rest_type'].str.contains(rest_type, na=False)]
    
    if len(dff) == 0:
        return html.P("No restaurants found with these filters.", style={'color':'gray'})

    # Best cuisine by avg rating + vote count
    cuisine_stats = (dff.assign(cuisine=dff['cuisines'].str.split(','))
                     .explode('cuisine')
                     .assign(cuisine=lambda x: x['cuisine'].str.strip()))
    cuisine_stats = cuisine_stats.groupby('cuisine').agg(
        avg_rating=('rate','mean'),
        total_votes=('votes','sum'),
        count=('name','count'),
        high_perf=('high_performer','sum')
    ).reset_index()
    cuisine_stats = cuisine_stats[cuisine_stats['count'] >= 3].sort_values(
        'avg_rating', ascending=False).head(5)

    # Top performing restaurants
    top_rests = (dff.sort_values(['rate','votes'], ascending=False)
               .drop_duplicates(subset=['name'])
               .head(5))

    # Key stats
    avg_rating = dff['rate'].mean()
    hp_pct = dff['high_performer'].mean()*100
    competition = len(dff)

    return dbc.Row([
        # Market overview
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5(f"📊 Market Overview: {location}", style={'color':'#FF6B6B'}),
            html.P(f"Restaurants in budget: {competition}", style={'color':'white'}),
            html.P(f"Avg rating in area: {avg_rating:.2f} ⭐", style={'color':'white'}),
            html.P(f"High performer rate: {hp_pct:.1f}%", style={'color':'white'}),
            html.Hr(style={'borderColor':'#444'}),
            html.H6("🏆 Best Cuisines to Open:", style={'color':'#4ECDC4'}),
            html.Ul([html.Li(
                f"{row['cuisine']} — {row['avg_rating']:.2f}⭐ avg, {int(row['high_perf'])} high performers",
                style={'color':'white'}) 
                for _, row in cuisine_stats.iterrows()])
        ]), style={'backgroundColor':'#2a2a2a','height':'100%'}), width=5),

        # Top competitors
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5(f"🔍 Top Competitors to Study:", style={'color':'#FF6B6B'}),
            html.Table([
                html.Thead(html.Tr([
                    html.Th("Restaurant", style={'color':'#4ECDC4'}),
                    html.Th("Rating", style={'color':'#4ECDC4'}),
                    html.Th("Votes", style={'color':'#4ECDC4'}),
                    html.Th("Cost", style={'color':'#4ECDC4'}),
                ])),
                html.Tbody([html.Tr([
                    html.Td(row['name'][:25], style={'color':'white'}),
                    html.Td(f"{row['rate']}⭐", style={'color':'white'}),
                    html.Td(str(row['votes']), style={'color':'white'}),
                    html.Td(f"₹{int(row['cost'])}", style={'color':'white'}),
                ]) for _, row in top_rests.iterrows()])
            ], style={'width':'100%'})
        ]), style={'backgroundColor':'#2a2a2a','height':'100%'}), width=4),

        # Actionable insight
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("🎯 Key Insight:", style={'color':'#FF6B6B'}),
            html.P(
                f"In {location} under ₹{budget}, "
                f"'{cuisine_stats.iloc[0]['cuisine'] if len(cuisine_stats)>0 else 'N/A'}' "
                f"has the highest success rate. "
                f"Only {hp_pct:.0f}% of restaurants here are high performers — "
                f"{'low competition, good opportunity!' if hp_pct < 20 else 'competitive market, differentiate carefully.'}",
                style={'color':'white','fontSize':'15px'}),
            html.Hr(style={'borderColor':'#444'}),
            html.P("💡 Pro tip: Table booking feature correlates with +0.52 avg rating increase.",
                style={'color':'#96CEB4','fontStyle':'italic'}),
        ]), style={'backgroundColor':'#2a2a2a','height':'100%'}), width=3),
    ])

if __name__ == '__main__':
    app.run(debug=True)