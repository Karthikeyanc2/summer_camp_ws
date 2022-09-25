import numpy as np
import matplotlib
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.spatial import distance
np.random.seed(2)

original_num_classes = 3
num_points_per_class = 200
max_distance_to_move_the_center = 15


class KMeans:
    def __init__(self):
        self.labels = np.ones(original_num_classes * num_points_per_class).astype(int)
        self.points = np.random.randn(num_points_per_class * original_num_classes, 2)

        for i in range(original_num_classes):
            self.points[i * num_points_per_class:(i+1) * num_points_per_class, :] = \
                self.points[i * num_points_per_class:(i+1) * num_points_per_class, :] + \
                np.random.random(2) * max_distance_to_move_the_center - max_distance_to_move_the_center/2
            self.labels[i * num_points_per_class:(i+1) * num_points_per_class] = i

        self.fig = plt.figure(figsize=(16, 7.5))
        self.grid = GridSpec(56, 198, self.fig)
        self.ax_main = self.fig.add_subplot(self.grid[1:-1, 1:86])
        self.ax_main.axis('off')
        self.ax_main.set_title('K-means clustering')
        self.ax_main.set_aspect(1)
        self.prev_main_scatter = None
        self.prev_centers_scatter = None

        ax_b1 = self.fig.add_subplot(self.grid[1:3, 88:88 + 20])
        ax_b2 = self.fig.add_subplot(self.grid[1:3, 110:110 + 20])
        ax_b3 = self.fig.add_subplot(self.grid[1:3, 132:132 + 20])
        ax_b4 = self.fig.add_subplot(self.grid[1:3, 154:154 + 20])
        ax_b5 = self.fig.add_subplot(self.grid[1:3, 176:176 + 20])

        b1 = Button(ax_b1, "1 class")
        b1.on_clicked(lambda _: self.run_kmeans(num_classes=1))
        b2 = Button(ax_b2, "2 classes")
        b2.on_clicked(lambda _: self.run_kmeans(num_classes=2))
        b3 = Button(ax_b3, "3 classes")
        b3.on_clicked(lambda _: self.run_kmeans(num_classes=3))
        b4 = Button(ax_b4, "4 classes")
        b4.on_clicked(lambda _: self.run_kmeans(num_classes=4))
        b5 = Button(ax_b5, "5 classes")
        b5.on_clicked(lambda _: self.run_kmeans(num_classes=5))

        ax_b1 = self.fig.add_subplot(self.grid[5:27, 88:88 + 20])
        ax_b2 = self.fig.add_subplot(self.grid[5:27, 110:110 + 20])
        ax_b3 = self.fig.add_subplot(self.grid[5:27, 132:132 + 20])
        ax_b4 = self.fig.add_subplot(self.grid[5:27, 154:154 + 20])
        ax_b5 = self.fig.add_subplot(self.grid[5:27, 176:176 + 20])
        self.trials_axis = [ax_b1, ax_b2, ax_b3, ax_b4, ax_b5]
        self.trials_axis_plots = [None for _ in range(5)]

        for ax in self.trials_axis:
            ax.set_ylabel('average distance to center', labelpad=-2)
            ax.set_xlabel('trials', labelpad=-2)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])

        self.all_trial_results = [None, None, None, None, None]

        self.ax_elbow = self.fig.add_subplot(self.grid[29:-1, 88:-1])
        self.ax_elbow.set_ylabel('average distance to center', labelpad=-2)
        self.ax_elbow.set_xlabel('number of classes', labelpad=-2)
        self.ax_elbow.set_yticklabels([])
        self.ax_elbow.set_xticklabels([])
        self.ax_elbow.set_xticks([])
        self.ax_elbow.set_yticks([])

        self.results_for_elbow_plot = np.array([range(1, 6), np.zeros(5)]).T
        self.elbow_plot = []

        self.fig.tight_layout()
        plt.show()

    def run_kmeans(self, num_classes):
        results = []
        num_trials = 5
        for trial_num in range(num_trials):
            # print(f"trial num: {trial_num}; num_classes: {num_classes}")
            current_labels = np.random.randint(0, num_classes, len(self.points))
            prev_classes = current_labels

            for _ in range(50):
                if self.prev_main_scatter is not None:
                    self.prev_main_scatter.remove()
                if self.prev_centers_scatter is not None:
                    self.prev_centers_scatter.remove()

                self.prev_main_scatter = self.ax_main.scatter(self.points[:, 0], self.points[:, 1], color=cm.rainbow(current_labels / num_classes), marker='.', s=5)

                current_labels, current_centers = self.get_new_classes_and_centers(self.points, current_labels, num_classes)
                self.prev_centers_scatter = self.ax_main.scatter(current_centers[:, 0], current_centers[:, 1], marker='x', s=50, color='red', linewidths=2)
                if (prev_classes == current_labels).all():
                    break

                prev_classes = current_labels
                plt.pause(0.1)

            distances = np.linalg.norm(self.points[:, None, :] - current_centers[None, :, :], axis=2)
            results.append(np.min(distances, axis=1).mean())

            plt.pause(0.1)
        self.all_trial_results[num_classes-1] = results
        colors = ['red' for _ in range(num_trials)]
        colors[np.argmin(results)] = 'green'
        if self.trials_axis_plots[num_classes-1] is not None:
            self.trials_axis_plots[num_classes-1].remove()
        self.trials_axis_plots[num_classes-1] = self.trials_axis[num_classes-1].scatter(range(num_trials), results, c=colors, marker='x')

        self.results_for_elbow_plot[num_classes-1][1] = min(results)
        for plot in self.elbow_plot:
            plot.remove()
        self.elbow_plot = self.ax_elbow.plot(self.results_for_elbow_plot[:, 0], self.results_for_elbow_plot[:, 1])
        self.fig.canvas.draw()

    @staticmethod
    def get_new_classes_and_centers(points, current_labels, n_classes):
        """
        @param points: N x 2 (numpy array)
        @param current_labels: N (can take a number between 0 to n_classes-1)
        @param n_classes is a number
        @return new_labels (N), new_centers (n_classes x 2)
        """
        """
        How to find pair wise distances ?
        pair_wise_distances = distance.cdist(points_set_1, points_set_2)
        
        How to find the index of the minimum number ? 
        index_of_the_minimum_number = np.argmin([2,3,4,1,3])
        """
        return current_labels, np.zeros((n_classes, 2))


KMeans()
