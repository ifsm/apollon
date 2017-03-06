class PitchClassSom:
    def __init__(self, dims=(10, 10, 3), eta=.8, nh=5, metric='eucledian'):
        """
        Parameter:
            dims:    tripel (nx, ny, nw), mit
                     nx == Anzahl der Neuronen auf der x-Achse
                     ny == Anzahl der Neuronen auf der y-Achse
                     nw == Dimension der Feature-Vektoren

            eta:     Startwert für die Lernrate

            nh:      Startwert für den Nachbarschaftsradius

            metric:  Name der verwendeten Metrik


        Wir modellieren die Karte zwei-dimensionale Array, wobei die Zeilen die Neuronen
        darstellen und die Spalten die einzelnen Elemente des dazu gehörenden
        Gewichtsvektors. Beispiel:

                 [.....
                   w1  w2  w3
        Neuron 1  [.5, .6, .9],
        Neuron 2  [.3, .2, .1]
                 .......]

        Zusätzlich sollen die Gewichte automatische anhand einer uniformen
        Verteilung initialisiert werden.
        """
        # wir seichern die eingegebenen Dimensionen als Eigenschaft (member) des SOM
        self.shape = dims
        self.center = dims[0]//2, dims[1]//2

        self.init_eta = eta
        self.init_rnh = nh

        self.n_N = self.shape[0] * self.shape[1]    # Hilfsvariable für die Anzahl der Neuronen

        # Initialisierung der Karte
        self.lattice = np.random.uniform(0, 1, size=(self.n_N, self.shape[2]))

        # Initialisierung der activation map
        self._activation_map = np.zeros(self.n_N)

        # grid data
        self.grid = dstack(mgrid[0:dims[0], 0:dims[1]])


    def linear_decrease(self, sw, c_it, N_it):
        """Linearer Abfall

        Params:
            sw:     Startwert
            c_it:   aktuelle Iteration
            N_Nit:  Gesamtzahl der Interationen
        """
        return sw - sw * c_it / N_it

    def exponential_decrease(self, sw, c_it, N_it, slope=1):
        """Exponentieller Abfall.

        Params:
            sw:     Startwert
            c_it:   aktuelle Iteration
            N_Nit:  Gesamtzahl der Interationen
            slope:  Kontrolliert die Steilheit der Kurve
        """
        return np.exp(-c_it * slope / N_it) * sw

    def eucledian_distance(self, vA, vB):
        """Wir gehen davon aus, dass vA und vB numpy arrays sind."""
        return np.sqrt(((vA-vB)**2).sum())

    def plot_activation_map(self):
        plt.imshow(self._activation_map.reshape(self.shape[0], self.shape[1]),
                   vmin=0, cmap='Greys',
                   interpolation='None')
        if not plt.isinteractive():
            plt.show()

    def get_winners(self, data):
        return np.argmin(distance.cdist(data, self.lattice), axis=1)

    def neighbourhood(self, point, r_nh):
        var = stats.multivariate_normal(mean=point, cov=((r_nh, 0), (0, r_nh)))
        out = var.pdf(self.grid)
        return (out / np.max(out)).reshape(self.n_N, 1)

    def train(self, data, N_it):
        for i in range(N_it):
            c_eta = self.linear_decrease(self.init_eta, i, N_it)
            c_rnh = self.exponential_decrease(self.init_rnh, i, N_it)

            # get bmus
            bm_units = self.get_winners(data)

            # update activation map
            self._activation_map[bm_units] += 1


            b_idx = zip(*unravel_index(bm_units, (self.shape[0], self.shape[1])))

            for bi, di in zip(b_idx, data):
                c_nh = self.neighbourhood(bi, c_rnh)
                self.lattice += c_eta * c_nh * (di - self.lattice)

    def calibrate(self, data):
        bmu = self.get_winners(data)
        return unravel_index(bmu, (self.shape[0], self.shape[1]))

    def map_response(self, data_i):
        return distance.cdist(data_i[None, :], self.lattice)
