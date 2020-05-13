import csv
from abc import ABCMeta, abstractmethod
from functools import lru_cache

from numpy import array, zeros, linalg, dot, eye
from numpy.linalg import matrix_power


class Markov(metaclass=ABCMeta):

    matrix: array

    def __init__(self, matrix: array):
        self.matrix = matrix

    @classmethod
    def read_from_file(cls, filename: str) -> "Markov":
        """
        Открывает CVS файл и преобразует в array
        :param filename:
        :return:
        """
        with open(filename, encoding="utf-8") as file:
            return cls(array([list(map(float, row)) for row in csv.reader(file)]))

    @abstractmethod
    def probability_of_switching_to_state(self, of: int, to: int, n: int):
        pass

    @abstractmethod
    def probabilities_of_system_states_in_n_steps(
        self, n: int, start_step: array
    ) -> array:
        pass

    @abstractmethod
    def make_step_in_matrix(self, matrix: array) -> array:
        pass

    @abstractmethod
    def probability_of_the_first_transition(self, of: int, to: int, n: int) -> float:
        pass

    @abstractmethod
    def probability_of_transition_no_later_than(
        self, of: int, to: int, n: int
    ) -> float:
        pass

    @abstractmethod
    def average_number_of_steps_for_the_transition(self, of: int, to: int) -> float:
        pass

    @abstractmethod
    def probability_of_first_return(self, of: int, n: int) -> float:
        pass

    @abstractmethod
    def probability_of_return_no_later_than(self, of: int, n: int) -> float:
        pass

    @abstractmethod
    def average_time_of_return(self, of: int) -> float:
        pass

    @abstractmethod
    def steady_state_probability(self):
        pass


class MarkovProcess(Markov):
    def probability_of_switching_to_state(self, of: int, to: int, n: int) -> float:
        """
        Вероятность перехода в состояние
        
        :param of: 
        :param to: 
        :param n: 
        :return: 
        """
        return matrix_power(self.matrix, n)[of - 1, to - 1]

    def probabilities_of_system_states_in_n_steps(
        self, n: int, start_step: array
    ) -> array:
        """
        Вероятности состояний системы через n шагов с начальными данными

        :param n:
        :param start_step:
        :return:
        """

        return dot(matrix_power(self.matrix, n), start_step)

    def make_step_in_matrix(self, matrix: array) -> array:
        """
        Совершает переход

        :param matrix: матрица
        :return:
        """

        temp_matrix = zeros(self.matrix.shape)
        for i in range(self.matrix.shape[0]):
            for j in range(self.matrix.shape[1]):
                temp_matrix[i, j] = sum(
                    self.matrix[i, m] * matrix[m, j] if m != j else 0
                    for m in range(self.matrix.shape[0])
                )
        return temp_matrix

    def probability_of_the_first_transition(self, of: int, to: int, n: int) -> float:
        """
        Вероятность первого перехода

        :param of:
        :param to:
        :param n:
        :return:
        """

        temp_matrix = self.matrix
        for _ in range(2, n + 1):
            temp_matrix = self.make_step_in_matrix(temp_matrix)
        return temp_matrix[of - 1, to - 1]

    def probability_of_transition_no_later_than(
        self, of: int, to: int, n: int
    ) -> float:
        """
        Вероятность перехода не позднее чем

        :param of:
        :param to:
        :param n:
        :return:
        """

        return sum(
            self.probability_of_the_first_transition(of - 1, to - 1, step)
            for step in range(1, n + 1)
        )

    def average_number_of_steps_for_the_transition(self, of: int, to: int) -> float:
        """
        Среднее количество шагов для перехода

        :param of:
        :param to:
        :return:
        """

        of, to = map(lambda x: x - 1, (of, to))
        temp_matrix, count = self.matrix, self.matrix[of, to]
        for i in range(2, 1000):
            temp_matrix = self.make_step_in_matrix(temp_matrix)
            count += i * temp_matrix[of, to]
        return count

    @lru_cache(maxsize=None)
    def probability_of_first_return(self, of: int, n: int) -> float:
        """
        Вероятность первого возвращения

        :param of:
        :param n:
        :return:
        """

        temp_matrix = zeros(self.matrix.shape)
        for i in range(1, n):
            temp_matrix += self.probability_of_first_return(of, i) * matrix_power(
                self.matrix, n - i
            )
        return (matrix_power(self.matrix, n) - temp_matrix)[of - 1, of - 1]

    def probability_of_return_no_later_than(self, of: int, n: int) -> float:
        """
        Вероятность возвращения не позднее чем

        :param of:
        :param n:
        :return:
        """

        return sum(
            self.probability_of_first_return(of - 1, step) for step in range(1, n + 1)
        )

    def average_time_of_return(self, of: int) -> float:
        """
        Среднее время возвращения

        :param of:
        :return:
        """

        return sum(
            i * self.probability_of_first_return(of - 1, i) for i in range(1, 130)
        )

    def steady_state_probability(self) -> array:
        """
        Установившиеся вероятности

        :return:
        """

        m = self.matrix.T - eye(self.matrix.shape[0])
        m[-1, :] = 1
        return dot(linalg.inv(m), array([0] * (self.matrix.shape[0] - 1) + [1]))


def main():
    markov_process: MarkovProcess = MarkovProcess.read_from_file("matrix.csv")

    task_1 = markov_process.probability_of_switching_to_state(4, 5, 7)
    print(
        f"1. Вероятность того, что за 7 шагов система перейдет из состояния 4 в состояние 1: {task_1}"
    )

    task_2 = markov_process.probabilities_of_system_states_in_n_steps(
        10,
        array(
            (
                0.08,
                0.1,
                0.13,
                0.09,
                0.08,
                0.01,
                0.1,
                0.02,
                0.12,
                0.05,
                0.09,
                0.05,
                0.06,
                0.02,
            )
        ),
    )
    print(
        f"2. Вероятности состояний системы спустя 10 шагов с заданными начальными вероятностями: {', '.join(map(str, task_2))}"
    )

    task_3 = markov_process.probability_of_the_first_transition(3, 6, 7)
    print(
        f"3. Вероятность первого перехода за 7 шагов из состояния 3 в состояние 6: {task_3}"
    )

    task_4 = markov_process.probability_of_transition_no_later_than(14, 5, 8)
    print(
        f"4. Вероятность перехода из состояния 14 в состояние 5 не позднее чем за 8 шагов: {task_4}"
    )

    task_5 = markov_process.average_number_of_steps_for_the_transition(7, 1)
    print(
        f"5. Среднее количество шагов для перехода из состояния 7 в состояние 1: {task_5}"
    )

    task_6 = markov_process.probability_of_first_return(8, 10)
    print(f"6. Вероятность первого возвращения в состояние 8 за 10 шагов: {task_6}")

    task_7 = markov_process.probability_of_return_no_later_than(1, 9)
    print(
        f"7. Вероятность возвращения в состояние 1 не позднее чем за 9 шагов {task_7}"
    )

    task_8 = markov_process.average_time_of_return(10)
    print(f"8. Среднее время возвращения в состояние 10 {task_8}")

    task_9 = markov_process.steady_state_probability()
    print(f"9. Установившиеся вероятности: {', '.join(map(str, task_9))}")


if __name__ == "__main__":
    main()
