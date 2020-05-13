import numpy as np


class QueueingSystem:
    matrix: np.array
    stable_condition: np.array

    def __init__(self, arriving: int, channels: int, intensity: int, max_size: int):
        self.arriving = arriving
        self.channels = channels
        self.intensity = intensity
        self.max_size = max_size
        self.matrix = self.creation_of_transition_matrix(
            arriving, channels, intensity, max_size
        )
        self.stable_condition = self.establishment_of_the_steady_state_probabilities(
            self.matrix
        )

    @staticmethod
    def creation_of_transition_matrix(
        arriving: int, channels: int, intensity: int, max_size: int
    ) -> np.array:
        """
        Создание матрицы переходов

        :param arriving:
        :param channels:
        :param intensity:
        :param max_size:
        :return:
        """

        matrix = np.zeros((channels + max_size + 1, channels + max_size + 1))
        for i in range(channels + max_size):
            matrix[i, i + 1] = arriving
            matrix[i + 1, i] = (
                intensity * (i + 1) if i < channels else intensity * channels
            )
        return matrix

    @staticmethod
    def establishment_of_the_steady_state_probabilities(matrix: np.array) -> np.array:
        """
        Установившиеся вероятности

        :param matrix:
        :return:
        """

        new = matrix.T - np.diag([matrix[i, :].sum() for i in range(matrix.shape[0])])
        new[-1, :] = 1
        zeros = np.zeros(new.shape[0])
        zeros[-1] = 1
        return np.linalg.inv(new).dot(zeros)

    def probability_of_denial_of_service(self) -> float:
        """
        Вероятность отказа в обслуживании

        :return:
        """

        return self.stable_condition[-1]

    def bandwidth(self, absolute: bool = False) -> float:
        """
        Пропускная способность

        :param absolute:
        :return:
        """

        return (
            1 - self.stable_condition[-1] * self.arriving
            if absolute
            else 1 - self.stable_condition[-1]
        )

    def average_length_of_the_queue(self) -> float:
        """
        Средняя длина очереди

        :return:
        """

        return sum(
            i * self.stable_condition[self.channels + i]
            for i in range(1, self.max_size + 1)
        )

    def average_time_in_the_queue(self) -> float:
        """
        Среднее время в очереди

        :return:
        """

        return sum(
            (i + 1)
            / (self.channels * self.intensity)
            * self.stable_condition[self.channels + i]
            for i in range(self.max_size)
        )

    def average_number_of_busy_channels(self) -> float:
        """
        Среднее число занятых каналов

        :return:
        """

        return sum(
            i * self.stable_condition[i] for i in range(1, self.channels + 1)
        ) + sum(
            self.channels * self.stable_condition[i]
            for i in range(self.channels + 1, self.channels + self.max_size + 1)
        )

    def probability_that_the_incoming_request_will_not_wait_in_the_queue(self) -> float:
        """
        Вероятность что поступающая заявка не будет ждать в очереди

        :return:
        """

        return sum(self.stable_condition[: self.channels])

    def average_downtime(self) -> float:
        """
        Среднее время простоя

        :return:
        """

        return 1 / self.arriving

    def average_time_when_there_is_no_queue_in_the_system(self) -> float:
        """
        Среднее время, когда в системе нет очереди

        :return:
        """

        probabilities = {i: self.stable_condition[i] for i in range(self.channels + 1)}
        normal_coefficient = 1 / sum(probabilities.values())
        probabilities = {k: v * normal_coefficient for k, v in probabilities.items()}
        return self.time_spent_in_a_subset(
            set(probabilities.keys()), probabilities, 0.0005, 10
        )

    def time_spent_in_a_subset(
        self, subset: set, probabilities: dict, time_step: float, time: int
    ) -> float:
        """
        Время пребывания в подмножестве

        :param subset: список состояний
        :param probabilities: вероятности
        :param time_step: шаг времени
        :param time: полное время проведения испытаний
        :return: время нахождения
        """

        buffer_array = np.zeros(self.matrix.shape)
        for i in subset:
            for j in range(self.matrix.shape[0]):
                buffer_array[j][i] = self.matrix[i][j]
        for i in subset:
            for j in range(self.matrix.shape[0]):
                if j != i:
                    buffer_array[i][i] -= self.matrix[i][j]

        not_subset, f = set(range(self.matrix.shape[0])) - subset, 0
        for t in np.arange(time_step, time + time_step, time_step):
            probabilities = self.probability_search(
                buffer_array, subset, probabilities, time_step
            )
            f += t * sum(
                probabilities[i] * self.matrix[i][j] for j in not_subset for i in subset
            )
        return f * time_step

    @staticmethod
    def probability_search(
        buffer_array: np.ndarray, subset: set, probabilities: list, time: float
    ) -> list:
        """
        Поиск вероятности

        :param buffer_array:
        :param subset:
        :param probabilities:
        :param time:
        :return:
        """

        coefficients = {x: [0, 0, 0, 0] for x in subset}
        for i in subset:
            for j in subset:
                coefficients[i][0] += buffer_array[i][j] * probabilities[j]
            for j in subset:
                coefficients[i][1] += (buffer_array[i][j]) * (
                    probabilities[j] + coefficients[i][0] * time / 2
                )
            for j in subset:
                coefficients[i][2] += (buffer_array[i][j]) * (
                    probabilities[j] + coefficients[i][1] * time / 2
                )
            for j in subset:
                coefficients[i][3] += (buffer_array[i][j]) * (
                    probabilities[j] + coefficients[i][2] * time
                )
        return [
            (
                probabilities[x]
                + time
                * (
                    coefficients[x][0]
                    + 2 * coefficients[x][1]
                    + 2 * coefficients[x][2]
                    + coefficients[x][3]
                )
                / 6
            )
            for x in subset
        ]


def main():
    queueing_system: QueueingSystem = QueueingSystem(45, 5, 12, 15)
    print(
        f"a) Установившиеся вероятности: {', '.join(map(str, queueing_system.stable_condition))}"
    )
    print(
        f"b) Вероятность отказа в обслуживании: {queueing_system.probability_of_denial_of_service()}"
    )
    print(
        f"c) Относительная/абсолютная интенсивность обслуживания:"
        f" {queueing_system.bandwidth()} / {queueing_system.bandwidth(absolute=True)}"
    )
    print(f"d) Средняя длина очереди: {queueing_system.average_length_of_the_queue()}")
    print(f"e) Среднее время в очереди: {queueing_system.average_time_in_the_queue()}")
    print(
        f"f) Среднее число занятых каналов: {queueing_system.average_number_of_busy_channels()}"
    )
    print(
        f"g) Вероятность того, что поступающая заявка не будет ждать в очереди:"
        f" {queueing_system.probability_that_the_incoming_request_will_not_wait_in_the_queue()} "
    )
    print(
        f"h) Среднее время простоя системы массового обслуживания: {queueing_system.average_downtime()}"
    )
    print(
        f"i) Среднее время, когда в системе нет очереди:"
        f" {queueing_system.average_time_when_there_is_no_queue_in_the_system()}"
    )


if __name__ == "__main__":
    main()
