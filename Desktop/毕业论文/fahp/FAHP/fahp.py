import pandas as pd
from functools import reduce
import numpy as np


class FuzzyNum:
    def __init__(self):
        self.data_frame = pd.read_csv("count.csv")
        self.data_frame_rows = self.data_frame.shape[0]
        self.data_frame_columns = self.data_frame.shape[1]

    @staticmethod
    def find_comma(strings):
        comma = ","
        first_comma_place = strings.find(comma)
        second_comma_place = strings.find(comma, first_comma_place+1)
        return first_comma_place, second_comma_place

    def convert_to_num(self):
        upper = []
        middle = []
        less = []
        for row in self.data_frame.values:
            for fuzzy_num in row:
                vector = fuzzy_num
                fc, sc = self.find_comma(vector)
                upper.append(float(eval(vector[:fc])))
                middle.append(float(eval(vector[fc+1:sc])))
                less.append(float(eval(vector[sc+1:])))
        fuzzy_matrix = [upper, middle, less]
        matrix = np.array(fuzzy_matrix)
        return matrix

    # 计算整个模糊矩阵的总值
    def calculator_all_matrix(self):
        matrix = self.convert_to_num()
        fuzzy_matrix = [matrix[0].sum(), matrix[1].sum(), matrix[2].sum()]
        return fuzzy_matrix

    # 按行计算模糊矩阵的值
    def calculator_matrix_row(self):
        matrix_sum = self.calculator_all_matrix()
        rows = self.data_frame_rows
        columns = self.data_frame_columns
        matrix = self.convert_to_num()
        less = matrix[0].reshape(rows, columns).sum(axis=1)/matrix_sum[2]
        middle = matrix[1].reshape(rows, columns).sum(axis=1)/matrix_sum[1]
        upper = matrix[2].reshape(rows, columns).sum(axis=1)/matrix_sum[0]
        return less, middle, upper

    # 重新按各因素封装模糊矩阵的返回矩阵
    def reshape_matrix(self):
        l, m, u = self.calculator_matrix_row()
        new_matrix = []
        for i in range(len(l)):
            a = l[i], m[i], u[i]
            vm = sorted(a)
            new_matrix.append(vm)
        return new_matrix

    @staticmethod
    def delete(index, vector):
        return vector[:index] + vector[index+1:]

    # 对模糊数去模糊化
    def remove_fuzzy(self):
        result = []
        c_result = []
        matrix = self.reshape_matrix()
        for index in range(len(matrix)):
            temp_matrix = self.delete(index, matrix)
            for vector in temp_matrix:
                result.append(self.get_c_number(matrix[index], vector))
        for i in range(1, self.data_frame_columns+1):
            c_result.append(result[(self.data_frame_columns-1) * (i-1):(self.data_frame_columns-1)*i])
        c_result = np.array(c_result)
        return c_result

    @staticmethod
    def get_c_number(now_vector, vector):
        if now_vector[1] >= vector[1]:
            return 1
        elif now_vector[2] >= vector[0]:
            return (now_vector[2] - vector[0]) / ((now_vector[2] - now_vector[1])+(vector[1] - vector[0]))
        else:
            return 0

    # 计算每个行向量的最小值，归一化最后输出
    def get_min_dc(self):
        matrix = self.remove_fuzzy()
        vector_min = matrix.min(axis=1)
        vector_sum = vector_min.sum()
        vector = vector_min/vector_sum
        return vector


if __name__ == '__main__':
    fn = FuzzyNum()
    print(fn.calculator_all_matrix())
    print("模糊权重值为： \n"+"*"*30)
    print(fn.get_min_dc().reshape(fn.data_frame_columns,1))
    print("*"*30)



