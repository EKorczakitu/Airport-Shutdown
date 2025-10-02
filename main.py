import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

airports = pd.read_csv('data/airports.csv')
routes = pd.read_csv('data/routes.csv')



#print(airports.columns)
#print(routes.columns)

airports_df = airports[["name", "type", "iso_country", "iata_code", "continent"]]
routes_df = routes[["Airline", "Source airport", "Destination airport"]]

#print(airports_df.head())
#print(routes_df.head())

print(airports_df.shape)
print(routes_df.shape)

print(airports_df.isnull().sum())
print(routes_df.isnull().sum())

airports_df = airports_df.dropna()
routes_df = routes_df.dropna()

print(airports_df.shape)
print(routes_df.shape)

print(airports_df["iso_country"].value_counts())