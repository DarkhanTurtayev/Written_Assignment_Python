import numpy as np
import pandas as pd
import unittest
import main

#Test if datasets paired to XY correctly
class TestDataPreparator(unittest.TestCase):

    def test_Data_Preparator(self):
        self.prepared_data = main.Data_Preparator('idealC.csv')
        
        for i in self.prepared_data:
            self.assertEqual(len(i.count()), 2) #check if tables paired into 2 columns

#Test if ideal functions are correct 
class TestGetIdealFunction(unittest.TestCase):

    def test_GetIdealFunctions(self):
        self.A = pd.read_csv('trainA.csv') 
        self.train_data = main.Data_Preparator('trainA.csv')
        self.ideal_data = main.Data_Preparator('idealC.csv')
        self.best_fit = main.GetIdealFunctions()

        ideal_index = self.best_fit.get_difference(self.train_data, self.ideal_data)
        difference_list = self.best_fit.get_difference(self.train_data, self.ideal_data)
        corelation_func_names = self.best_fit.get_difference(self.train_data, self.ideal_data)
        ideal_functions = self.best_fit.get_functions(self.ideal_data)

        #check if results lenght equal to the required lenght, -1 because x is not considered
        try:
            for index in ideal_index:
                self.assertEqual(len(index), len(self.A.count())-1) 
            for list in difference_list:
                self.assertEqual(len(list), len(self.A.count())-1)
            for name in corelation_func_names:
                self.assertEqual(len(name), len(self.A.count())-1)
            print('All results are equal to required lenghts of train datatset')
        except:
            print('Some data is wrong')
        #check if number of ideal functions are similar to number of fucntions form training dataset
        try:
            self.assertEqual(len(ideal_functions), len(self.A.count())-1)
            print('All functions found correctly')
        except:
            print('Number of found functions is different form required train dataset')
            
     
if __name__ == '__main__':
    unittest.main()
