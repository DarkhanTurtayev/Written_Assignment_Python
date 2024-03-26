import numpy as np
import pandas as pd
import sqlalchemy as db
from sqlalchemy import create_engine
import math
from bokeh.plotting import figure, show
from bokeh.layouts import gridplot
from bokeh.models import HoverTool

# import csv tables as Pandas Dataframes
ideal_C =pd.read_csv('idealC.csv') 
test_B =pd.read_csv('testB.csv') 
train_A =pd.read_csv('trainA.csv') 

# Testing functions are in separate test_main file.

'''
The process of selecting 4 ideal functions:
1) Reestablish csv tables into XY paired tables
2) Find least squares difference between ideal (C) and train (A) datasets
4) Compare the deviation between them to find functions with minimal ones
5) Create seperate list of ideal 4 functions 

The process for finding best fit points:
1) Remove of points that are not in x dimension of any given functions
2) Find the absolute vectors of point of test (B) dataset
3) Compare absolute vectors between points from test(B) and each of ideal functions table, 
check if distance between is not larger than factor of sqrt(2).
4)Plot everything and extract SQL databases 
'''

# Function for preparing csv table as iterable dataframe tables (XY pairs)
class Data_Preparator: # Intitialize each csv table into separate xy table pairs

    def __init__(self,data_path):
        self.separated_tables = []
        try:
            self.data_func = pd.read_csv(data_path)
        except Exception:
            print('No file found')
            raise

        for x, y in self.data_func.items():
            x_data = self.data_func['x']
            if not x_data is y:
                table = pd.concat([x_data, y],axis=1)
                self.separated_tables.append(table)
            else:
                continue
    #Make dataframe object iterable
    def __iter__(self): 
        return iter(self.separated_tables)

# Function to separate X and Y lists from the created dataframe tables 
class xy_separator:
    def __init__(self, each_table):
        df = pd.DataFrame(each_table)

        self.x_list = []
        self.y_list = []
        for i in range(len(df)): # x index 0
            y = df.iloc[i,1]
            x = df.iloc[i,0]
            self.x_list.append(x) #way to extract x values from the XY dataframes
            self.y_list.append(y) #way to extract y values from the XY dataframes

# Function to generate SQL file from given Dataframe
class MakeSQL:
    def __init__(self, data_set, name):
     try:
            connection = f"sqlite:///{name}.db"
            engine = create_engine(connection)
            table_name = name
            data_set.to_sql(table_name, con=engine, if_exists='replace', index=False)
            engine.dispose()
     except:
         print('Unable to save SQL file')
     finally:
         engine.dispose()


#(TASK 1) calculating 4 ideal functions: train(A) and ideal(C)
class GetIdealFunctions:
    def get_difference(self, train, ideal):
        self.ideal_index = [] # list of indexes of 4 ideal dunctions
        self.difference_list = [] # absolute values for maximum deviation within the ideal 4 functions
        self.corelation_func_names = []
        #Iterating through Y values in Train A Dataset
        for yA in train:
            y_A = np.array(xy_separator(yA).y_list)
            name_A = str(pd.DataFrame(yA).columns[1])
            
            lses = []
            mdevs = []
            dict_nameinx = []
            
            #Iterating through Y values in Ideal 50 C Dataset to calculate best fit sum squared difference and minimal vectoral diffrence in distance between points
            for yC in ideal:
                y_C = np.array(xy_separator(yC).y_list) #extract y_column and convert them to Numpy Array
                name_C = str(pd.DataFrame(yC).columns[1]) #extact names of columns
                
                difference = abs(y_A - y_C) #difference of each point
                difference_squared = (y_A - y_C) ** 2 #squared difference for given points
                lse = np.sum(difference_squared) #lse - least squared error 

                maxdev = np.max(difference) #maximum difference for each XY pair 
                mdevs.append(maxdev) 
                lses.append(lse) #appending list of least squred errors

                tuple_c = (name_C, lse)
                dict_nameinx.append(tuple_c)

                
            minlse = np.min(lses) #finding minimal erorrs
            index = np.argmin(lses) #getting names of the columns with minimal lse
            self.ideal_index.append(index)
            self.difference_list.append(mdevs[index])
        
            #loop to affiliate test function with selected ideal function by names
            for tup in dict_nameinx:
                if tup[1] == minlse:
                    tup_all = (name_A, tup[0])
                    self.corelation_func_names.append(tup_all)
                    


            
       
        return self.ideal_index , self.difference_list, self.corelation_func_names



    #Extracting 4 Ideal functions from 50 candidates using found indecies
    def get_functions(self,dataC):

        self.ideal_functions = []

        for index,xy in enumerate(dataC):
            for i in self.ideal_index:
                if i == index:
                    self.ideal_functions.append(xy)
                else:
                    continue
        return self.ideal_functions


#Classes to make plots using Bokeh
class Plot: 
    def __init__(self, datasetA, datasetC, names): 
        plots = []
        for i in names: #loop to match train and ideal functions
            ideal_column = datasetC[i[1]] 
            train_column = datasetA[i[0]]
            train_x = train_A['x']
            df_merge = pd.concat([train_x,train_column,ideal_column], axis=1)
            
            #creating the plots
            x_column = df_merge.columns[0]
            p = figure(title=f'Train {i[0]} (blue) to Ideal {i[1]} (red)', width=700, height=500, x_axis_label=x_column, y_axis_label='y')
            p.line(x=df_merge[x_column], y=df_merge.iloc[:,1],line_color='blue', line_width=3)
            p.line(x=df_merge[x_column], y=df_merge.iloc[:,2],line_color='red', line_width=2)
            plots.append(p) 
        
        grid = gridplot(plots, ncols=2, toolbar_location=None)

        show(grid)

class Plot_points: #Plotting class for putting best fit points over ideal function graphs
    def __init__(self, dataset, dataset2):
        plots = []
        
        for i in dataset:
            data = pd.DataFrame(i)
            name = data.columns[1]
            x_column = data.columns[0]
            p = figure(title=f'{name} function', width=700, height=500, x_axis_label=x_column, y_axis_label='y')
            p.line(x=data.iloc[:,0], y=data.iloc[:,1],line_color='blue', line_width=2)
    
            current_df = dataset2[dataset2['Func_Number'] == name]
            
            for x,y,delta in zip(current_df['X'], current_df['Y_Test'], current_df['Y_Delta']):
                points = p.circle(x,y,size = 5, color='red')

                #Implementing Hover function to make process of graph review easier
                hover = HoverTool(renderers = [points], tooltips=[("Delta", f'{delta}'), ("Coordinates", f'{x},{y}')])
                p.add_tools(hover)

            
            
            plots.append(p)
    
        grid = gridplot(plots, ncols=2, toolbar_location=None)

        show(grid)

                

# Preparing data to pass for BestFit finder
clean_data_A = Data_Preparator('trainA.csv')
clean_data_C = Data_Preparator('idealC.csv')
#MAIN initiation
if __name__ == '__main__':
   
    best_fit_init = GetIdealFunctions() #initiate class to get data about best 4 functions
    best_fit_input = best_fit_init.get_difference(train = clean_data_A, ideal = clean_data_C)
    deviations_4 = best_fit_init.difference_list #get the list of deviations for TASK 2
    ideal4_functions = best_fit_init.get_functions(dataC=clean_data_C) #get 4 ideal functions from 50 in dataset C 

    
    factor= math.sqrt(2) 
    results = []



    #(TASK 2)
    #Compare deviation * sqrt(2) on points in test dataset B

    for ideal, maxdev in zip(ideal4_functions, deviations_4):

        
        df4 = pd.DataFrame(ideal)
        y_name = df4.columns[1]
        dfB = pd.DataFrame(test_B).sort_values(by='x')     #Just in testing purposes sorting test B dataset from minimum to maximum X  
        merged_df = df4.merge(dfB, on='x',how='inner')     #Removing all XY pairs that are not in test B dataset based on X values 
        dfY_difference = pd.DataFrame(abs(merged_df.iloc[:,1] - merged_df.iloc[:,2])) #Calculating absolute vector disstance of each point 
        
        filtered_df = dfY_difference[dfY_difference.iloc[:,0]/maxdev <= factor] #Check if points within the right distance based on maximum deviation and factor of sqrt(2)
        
        full_data = pd.merge(merged_df, filtered_df,left_index=True, right_index=True)
        full_data['Func_n'] = y_name
        new_columns_names= ['X', 'Y_Ideal', 'Y_Test', 'Y_Delta', 'Func_Number']
        full_data.columns = new_columns_names

        results.append(full_data)


    final_results = pd.concat(results, axis=0)
    final_rounded_results = final_results.round(2)
    

    # Create SQL databases 
    try:
        SQLA = MakeSQL(train_A, 'Train Functions, DataSet A')
        SQLC = MakeSQL(ideal_C, 'Ideal 50 Functions, Dataset C')
        SQLB = MakeSQL(final_rounded_results, 'Test Dataset B and Delta Y corelations for 4 Ideal functions')
        print('Succesfully created databases')
    except:
        print('Unable to create SQL')
    
    #Get the names of the corelated functions to make the right plot
    indexes_names = []
    for i in ideal4_functions:
        df = pd.DataFrame(i)
        name = df.columns[1]
        indexes_names.append(name)
    
    
    corelated_functions = best_fit_init.corelation_func_names
    
    #Plotting the functions
    try:
        Plot(datasetA=train_A, datasetC=ideal_C, names=corelated_functions)
        print('Succesfully ploted functions')
    except:
        print('Unable to make a plot')
    try:
        Plot_points(dataset=ideal4_functions, dataset2=final_rounded_results)
        print('Succesfully ploted best fit points')
    except:
        print('Unable to make a plot of points')
        

    print(f'Names of Train functions and 4 Bestfit functions for them:\n {corelated_functions}')
    print(f'Here is the table for the test points on 4 ideal functions: \n {final_rounded_results}')
    print('INFO: In order to see further details on the graphs, please hover the cursor over preferable point')
    print('AUTHOR: Darkhan Turtayev')


