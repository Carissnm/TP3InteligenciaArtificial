import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * TP3 - Prototipo de identificación de imágenes con el modelo de Hopfield.
 */
public class Hopfield {

    // ------------------------------------------------------------------
    // Parámetros generales del prototipo
    // ------------------------------------------------------------------
    static final int LADO = 10;              // la imagen es de LADO x LADO
    static final int N = LADO * LADO;        // cantidad de neuronas
    static final int MAX_CICLOS = 20;        // corte de seguridad
    static final long SEMILLA = 2026;        // para que los resultados sean repetibles

    enum Metodo { HEBB, PSEUDOINVERSA }

    // ------------------------------------------------------------------
    // Estado de la red
    // ------------------------------------------------------------------
    private final double[][] w = new double[N][N];   // matriz de pesos
    private Metodo metodo;

    // ==================================================================
    // 1) ENTRENAMIENTO
    // ==================================================================

    /** Regla de Hebb: suma de productos externos de cada patrón, sin autoconexión. */
    void entrenarHebb(List<int[]> patrones) {
        metodo = Metodo.HEBB;
        limpiarPesos();
        for (int[] x : patrones) {
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    if (i != j) {            
                        w[i][j] += x[i] * x[j];
                    }
                }
            }
        }
    }

    /**
     * Pseudoinversa (Personnaz, 1986): W = U (U^T U)^-1 U^T.
     * U tiene un patrón por columna (N x q). Solo exige que los patrones
     * sean linealmente independientes (si no, U^T U es singular).
     */
    void entrenarPseudoinversa(List<int[]> patrones) {
        metodo = Metodo.PSEUDOINVERSA;
        int q = patrones.size();
        double[][] u = new double[N][q];
        for (int k = 0; k < q; k++) {
            for (int i = 0; i < N; i++) {
                u[i][k] = patrones.get(k)[i];
            }
        }
        double[][] ut = Algebra.transpuesta(u);                 // q x N
        double[][] utuInv = Algebra.inversa(Algebra.producto(ut, u)); // q x q
        double[][] pseudo = Algebra.producto(utuInv, ut);       // U† (q x N)
        double[][] proy = Algebra.producto(u, pseudo);          // U U† (N x N)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                // Se anula la diagonal: en Hopfield ninguna neurona se realimenta a sí misma.
                w[i][j] = (i == j) ? 0.0 : proy[i][j];
            }
        }
    }

    private void limpiarPesos() {
        for (double[] fila : w) {
            java.util.Arrays.fill(fila, 0.0);
        }
    }

    // ==================================================================
    // 2) CONSULTA (recuperación)
    // ==================================================================

    /** Función energía: E = -1/2 x^T W x  (umbrales nulos). */
    double energia(int[] x) {
        double e = 0;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                e += x[i] * w[i][j] * x[j];
            }
        }
        return -0.5 * e;
    }

    /**
     * Dinámica síncrona: en cada ciclo todas las neuronas calculan
     * y = signo(W x) a la vez, y la salida pasa a ser la nueva entrada.
     * Si el potencial es exactamente 0, la neurona conserva su estado.
     * Se detiene cuando la salida coincide con la entrada (equilibrio),
     * cuando detecta un ciclo de período 2 o al llegar a MAX_CICLOS.
     */
    Resultado consultar(int[] entrada) {
        Resultado r = new Resultado();
        int[] x = entrada.clone();
        int[] anterior = null;
        r.estados.add(x.clone());
        r.energias.add(energia(x));

        for (int ciclo = 1; ciclo <= MAX_CICLOS; ciclo++) {
            int[] y = new int[N];
            for (int i = 0; i < N; i++) {
                double u = 0;
                for (int j = 0; j < N; j++) {
                    u += w[i][j] * x[j];
                }
                y[i] = (u > 0) ? 1 : (u < 0) ? -1 : x[i];
            }
            if (iguales(y, x)) {                 // entrada = salida: equilibrio
                r.estable = true;
                break;
            }
            r.estados.add(y.clone());
            r.energias.add(energia(y));
            if (anterior != null && iguales(y, anterior)) {
                r.oscilacion = true;             // ciclo de período 2
                break;
            }
            anterior = x;
            x = y;
        }
        r.salida = r.estados.get(r.estados.size() - 1);
        return r;
    }

    static class Resultado {
        List<int[]> estados = new ArrayList<>();
        List<Double> energias = new ArrayList<>();
        int[] salida;
        boolean estable = false;
        boolean oscilacion = false;

        int ciclos() {
            return estados.size() - 1;
        }
    }

    // ==================================================================
    // 3) ÁLGEBRA MATRICIAL (programada a mano)
    // ==================================================================
    static class Algebra {

        static double[][] transpuesta(double[][] a) {
            double[][] t = new double[a[0].length][a.length];
            for (int i = 0; i < a.length; i++) {
                for (int j = 0; j < a[0].length; j++) {
                    t[j][i] = a[i][j];
                }
            }
            return t;
        }

        static double[][] producto(double[][] a, double[][] b) {
            int f = a.length, m = b.length, c = b[0].length;
            double[][] p = new double[f][c];
            for (int i = 0; i < f; i++) {
                for (int k = 0; k < m; k++) {
                    double aik = a[i][k];
                    if (aik == 0) continue;
                    for (int j = 0; j < c; j++) {
                        p[i][j] += aik * b[k][j];
                    }
                }
            }
            return p;
        }

        /** Inversa por Gauss-Jordan con pivoteo parcial. */
        static double[][] inversa(double[][] a) {
            int n = a.length;
            double[][] m = new double[n][2 * n];
            for (int i = 0; i < n; i++) {
                System.arraycopy(a[i], 0, m[i], 0, n);
                m[i][n + i] = 1.0;
            }
            for (int col = 0; col < n; col++) {
                int piv = col;
                for (int f = col + 1; f < n; f++) {
                    if (Math.abs(m[f][col]) > Math.abs(m[piv][col])) piv = f;
                }
                if (Math.abs(m[piv][col]) < 1e-9) {
                    throw new ArithmeticException(
                        "U^T U es singular: los patrones no son linealmente independientes");
                }
                double[] tmp = m[col]; m[col] = m[piv]; m[piv] = tmp;
                double d = m[col][col];
                for (int j = 0; j < 2 * n; j++) m[col][j] /= d;
                for (int f = 0; f < n; f++) {
                    if (f == col) continue;
                    double factor = m[f][col];
                    if (factor == 0) continue;
                    for (int j = 0; j < 2 * n; j++) m[f][j] -= factor * m[col][j];
                }
            }
            double[][] inv = new double[n][n];
            for (int i = 0; i < n; i++) {
                System.arraycopy(m[i], n, inv[i], 0, n);
            }
            return inv;
        }
    }

    // ==================================================================
    // 4) IMÁGENES: construcción, ruido, dibujo y medición
    // ==================================================================
    static class Imagen {

        /** Escuadra de referencia fija en el ángulo inferior izquierdo. */
        static final int[][] ESCUADRA = {{7, 0}, {8, 0}, {9, 0}, {9, 1}, {9, 2}};

        /** Aro de 6 x 6 con vértices redondeados (reemplaza al aro "C"). */
        static final String[] ARO = {
            "-XXXX-",
            "X----X",
            "X----X",
            "X----X",
            "X----X",
            "-XXXX-"
        };

        /** Figura distinta (cruz) que se usa como patrón "distractor". */
        static final String[] CRUZ = {
            "--XX--",
            "--XX--",
            "XXXXXX",
            "XXXXXX",
            "--XX--",
            "--XX--"
        };

        static int[] vacia() {
            int[] x = new int[N];
            java.util.Arrays.fill(x, -1);
            return x;
        }

        /** Imagen con la escuadra y una figura cuya esquina sup. izq. está en (fila, col). */
        static int[] conFigura(String[] figura, int fila, int col) {
            int[] x = vacia();
            for (int[] p : ESCUADRA) x[p[0] * LADO + p[1]] = 1;
            for (int i = 0; i < figura.length; i++) {
                for (int j = 0; j < figura[i].length(); j++) {
                    if (figura[i].charAt(j) == 'X') {
                        x[(fila + i) * LADO + (col + j)] = 1;
                    }
                }
            }
            return x;
        }

        /** Invierte un porcentaje de pixeles elegidos al azar (ruido). */
        static int[] conRuido(int[] x, double porcentaje, Random rnd) {
            int[] y = x.clone();
            int cant = (int) Math.round(N * porcentaje);
            List<Integer> idx = new ArrayList<>();
            for (int i = 0; i < N; i++) idx.add(i);
            java.util.Collections.shuffle(idx, rnd);
            for (int k = 0; k < cant; k++) y[idx.get(k)] = -y[idx.get(k)];
            return y;
        }

        /** Borra (apaga) un porcentaje de los píxeles activos de la figura (defectos). */
        static int[] conFaltantes(int[] x, double porcentaje, Random rnd) {
            int[] y = x.clone();
            List<Integer> activos = new ArrayList<>();
            for (int i = 0; i < N; i++) {
                if (y[i] == 1 && !esEscuadra(i)) activos.add(i);
            }
            java.util.Collections.shuffle(activos, rnd);
            int cant = (int) Math.round(activos.size() * porcentaje);
            for (int k = 0; k < cant; k++) y[activos.get(k)] = -1;
            return y;
        }

        static boolean esEscuadra(int idx) {
            for (int[] p : ESCUADRA) {
                if (p[0] * LADO + p[1] == idx) return true;
            }
            return false;
        }

        /** Verifica que la escuadra de referencia esté completa en la imagen. */
        static boolean escuadraPresente(int[] x) {
            for (int[] p : ESCUADRA) {
                if (x[p[0] * LADO + p[1]] != 1) return false;
            }
            return true;
        }

        /**
         * Centro "A" de la figura: promedio de las posiciones de los pixeles
         * activos que no pertenecen a la escuadra, medido desde el vértice
         * de la escuadra (fila 9, columna 0). Devuelve {X, Y} en píxeles:
         * X hacia la derecha, Y hacia arriba.
         */
        static double[] centro(int[] x) {
            double sx = 0, sy = 0;
            int c = 0;
            for (int i = 0; i < N; i++) {
                if (x[i] == 1 && !esEscuadra(i)) {
                    int fila = i / LADO, col = i % LADO;
                    sx += col;
                    sy += (LADO - 1) - fila;
                    c++;
                }
            }
            if (c == 0) return null;
            return new double[]{sx / c, sy / c};
        }

        static int diferencias(int[] a, int[] b) {
            int d = 0;
            for (int i = 0; i < N; i++) if (a[i] != b[i]) d++;
            return d;
        }

        /** Dibuja varias imágenes lado a lado, con título y energía. */
        static String ladoALado(List<int[]> imgs, List<String> titulos) {
            StringBuilder sb = new StringBuilder();
            int ancho = 2 * LADO + 3;
            for (String t : titulos) sb.append(String.format("%-" + ancho + "s", t));
            sb.append('\n');
            for (int f = 0; f < LADO; f++) {
                for (int[] img : imgs) {
                    StringBuilder linea = new StringBuilder();
                    for (int c = 0; c < LADO; c++) {
                        linea.append(img[f * LADO + c] == 1 ? 'X' : '-').append(' ');
                    }
                    sb.append(String.format("%-" + ancho + "s", linea.toString()));
                }
                sb.append('\n');
            }
            return sb.toString();
        }
    }

    // ==================================================================
    // 5) EXPERIMENTOS
    // ==================================================================
    static boolean iguales(int[] a, int[] b) {
        return java.util.Arrays.equals(a, b);
    }

    static int indicePatron(int[] x, List<int[]> patrones) {
        for (int k = 0; k < patrones.size(); k++) {
            if (iguales(x, patrones.get(k))) return k;
        }
        return -1;
    }

    /** Muestra la secuencia de ciclos de una consulta y su diagnóstico. */
    static void mostrarConsulta(PrintWriter out, String titulo, Hopfield red, int[] entrada,
                                List<int[]> patrones, List<String> nombres, int[] esperado) {
        Resultado r = red.consultar(entrada);
        out.println("--- " + titulo + " [" + red.metodo + "] ---");
        List<int[]> imgs = new ArrayList<>();
        List<String> tit = new ArrayList<>();
        for (int k = 0; k < r.estados.size(); k++) {
            imgs.add(r.estados.get(k));
            tit.add((k == 0 ? "Entrada" : "Ciclo " + k)
                    + String.format(" E=%.1f", r.energias.get(k)));
        }
        imgs.add(esperado);
        tit.add("Referencia");
        // Máximo 5 imágenes por renglón para que entre en el ancho de la hoja
        for (int desde = 0; desde < imgs.size(); desde += 5) {
            int hasta = Math.min(desde + 5, imgs.size());
            out.println(Imagen.ladoALado(imgs.subList(desde, hasta), tit.subList(desde, hasta)));
        }
        int k = indicePatron(r.salida, patrones);
        out.println("Ciclos: " + r.ciclos()
                + (r.estable ? " | Estado estable" : r.oscilacion ? " | OSCILA (período 2)" : " | Sin converger"));
        out.println("Píxeles distintos de la referencia: " + Imagen.diferencias(r.salida, esperado)
                + " de " + N);
        out.println("Patrón recuperado: " + (k >= 0 ? nombres.get(k) : "NINGUNO (estado espurio)"));
        out.println("Escuadra de referencia intacta: " + (Imagen.escuadraPresente(r.salida) ? "sí" : "no"));
        double[] c = Imagen.centro(r.salida);
        double[] cRef = Imagen.centro(esperado);
        if (c != null && k >= 0) {
            out.printf("Centro A medido: X = %.1f px, Y = %.1f px (referencia X = %.1f, Y = %.1f)%n",
                    c[0], c[1], cRef[0], cRef[1]);
        }
        out.println();
    }

    public static void main(String[] args) throws IOException {
        try (PrintWriter out = new PrintWriter(new FileWriter("salida.txt"))) {
            PrintWriter dual = new PrintWriter(new java.io.Writer() {
                public void write(char[] c, int o, int l) {
                    out.write(c, o, l);
                    System.out.print(new String(c, o, l));
                }
                public void flush() { out.flush(); System.out.flush(); }
                public void close() { }
            }, true);
            ejecutar(dual);
            dual.flush();
        }
    }

    static void ejecutar(PrintWriter out) {
        Random rnd = new Random(SEMILLA);
        out.println("=================================================================");
        out.println(" TP3 - Prototipo de Hopfield para identificación de imágenes");
        out.println(" Imagen 10 x 10 = 100 neuronas | estados {+1 = X, -1 = -}");
        out.println("=================================================================\n");

        // ---------------------------------------------------------------
        // EXPERIMENTO 1: aro y cruz (dos figuras distintas) + escuadra
        // ---------------------------------------------------------------
        int[] aro = Imagen.conFigura(Imagen.ARO, 1, 3);
        int[] cruz = Imagen.conFigura(Imagen.CRUZ, 1, 3);
        List<int[]> p1 = List.of(aro, cruz);
        List<String> n1 = List.of("ARO", "CRUZ");

        out.println("EXPERIMENTO 1: dos patrones almacenados (aro y cruz, ambos con la escuadra)");
        out.println("Entrada: aro con 40% de sus píxeles faltantes y 12% de ruido en toda la imagen.\n");
        int[] e1 = Imagen.conRuido(Imagen.conFaltantes(aro, 0.40, rnd), 0.12, rnd);
        for (Metodo m : Metodo.values()) {
            Hopfield red = new Hopfield();
            if (m == Metodo.HEBB) red.entrenarHebb(p1); else red.entrenarPseudoinversa(p1);
            mostrarConsulta(out, "Aro dañado", red, e1, p1, n1, aro);
        }

        // ---------------------------------------------------------------
        // EXPERIMENTO 2: el aro en 5 posiciones (desplazamientos de 1 px)
        // ---------------------------------------------------------------
        int[][] pos = {{1, 3}, {1, 2}, {1, 4}, {0, 3}, {2, 3}};
        String[] desc = {"ARO nominal", "ARO 1px izquierda", "ARO 1px derecha",
                         "ARO 1px arriba", "ARO 1px abajo"};
        List<int[]> p2 = new ArrayList<>();
        List<String> n2 = new ArrayList<>();
        for (int k = 0; k < pos.length; k++) {
            p2.add(Imagen.conFigura(Imagen.ARO, pos[k][0], pos[k][1]));
            n2.add(desc[k]);
        }
        out.println("EXPERIMENTO 2: cinco patrones = el mismo aro desplazado 1 píxel");
        out.println("(simula los corrimientos del block sobre la cinta; se busca saber DÓNDE está el aro)");
        out.printf("Ortogonalidad: producto escalar entre aro nominal y aro corrido 1px = %d (0 sería ortogonal)%n%n",
                producto(p2.get(0), p2.get(2)));
        int[] objetivo = p2.get(2);   // el block quedó corrido 1 px a la derecha
        int[] e2 = Imagen.conRuido(Imagen.conFaltantes(objetivo, 0.25, rnd), 0.08, rnd);
        for (Metodo m : Metodo.values()) {
            Hopfield red = new Hopfield();
            if (m == Metodo.HEBB) red.entrenarHebb(p2); else red.entrenarPseudoinversa(p2);
            mostrarConsulta(out, "Aro corrido a la derecha, dañado", red, e2, p2, n2, objetivo);
        }

        // ---------------------------------------------------------------
        // EXPERIMENTO 3: tolerancia al ruido (estadística)
        // ---------------------------------------------------------------
        out.println("EXPERIMENTO 3: tolerancia al ruido - % de consultas que recuperan el patrón correcto");
        out.println("(100 consultas por celda, ruido = % de píxeles invertidos al azar)\n");
        out.println(String.format("%-8s | %-22s | %-22s", "Ruido", "Exp.1 (aro/cruz)", "Exp.2 (5 aros corridos)"));
        out.println(String.format("%-8s | %-10s %-11s | %-10s %-11s", "", "Hebb", "Pseudoinv.", "Hebb", "Pseudoinv."));
        out.println("-".repeat(58));
        Hopfield h1 = new Hopfield(); h1.entrenarHebb(p1);
        Hopfield s1 = new Hopfield(); s1.entrenarPseudoinversa(p1);
        Hopfield h2 = new Hopfield(); h2.entrenarHebb(p2);
        Hopfield s2 = new Hopfield(); s2.entrenarPseudoinversa(p2);
        double[] ruidos = {0.0, 0.05, 0.10, 0.20, 0.30, 0.40};
        for (double ru : ruidos) {
            out.println(String.format("%-8s | %-10s %-11s | %-10s %-11s",
                    (int) (ru * 100) + "%",
                    tasa(h1, p1, ru, rnd) + "%", tasa(s1, p1, ru, rnd) + "%",
                    tasa(h2, p2, ru, rnd) + "%", tasa(s2, p2, ru, rnd) + "%"));
        }
        out.println();

        // ---------------------------------------------------------------
        // EXPERIMENTO 4: capacidad de almacenamiento (patrones al azar)
        // ---------------------------------------------------------------
        out.println("EXPERIMENTO 4: capacidad de almacenamiento con patrones al azar (n = 100)");
        out.println("Estable = el patrón presentado intacto no cambia.");
        out.println("Recupera = el patrón con 10% de ruido vuelve exactamente al original.");
        out.println("(promedio de 10 repeticiones por fila)\n");
        out.println(String.format("%-5s %-6s | %-9s %-9s | %-9s %-9s", "q", "q/n",
                "Hebb", "", "Pseudoinv.", ""));
        out.println(String.format("%-5s %-6s | %-9s %-9s | %-9s %-9s", "", "",
                "Estable", "Recupera", "Estable", "Recupera"));
        out.println("-".repeat(56));
        int[] qs = {5, 10, 14, 20, 30, 40, 50, 60, 80};
        for (int q : qs) {
            int estH = 0, recH = 0, estP = 0, recP = 0, total = 0;
            for (int rep = 0; rep < 10; rep++) {
                List<int[]> pats = new ArrayList<>();
                for (int k = 0; k < q; k++) {
                    int[] x = new int[N];
                    for (int i = 0; i < N; i++) x[i] = rnd.nextBoolean() ? 1 : -1;
                    pats.add(x);
                }
                Hopfield hh = new Hopfield(); hh.entrenarHebb(pats);
                Hopfield pp = new Hopfield(); pp.entrenarPseudoinversa(pats);
                for (int[] x : pats) {
                    int[] ruidoso = Imagen.conRuido(x, 0.10, rnd);
                    if (iguales(hh.consultar(x).salida, x)) estH++;
                    if (iguales(hh.consultar(ruidoso).salida, x)) recH++;
                    if (iguales(pp.consultar(x).salida, x)) estP++;
                    if (iguales(pp.consultar(ruidoso).salida, x)) recP++;
                    total++;
                }
            }
            out.println(String.format("%-5d %-6.2f | %-9s %-9s | %-9s %-9s", q, q / (double) N,
                    pct(estH, total), pct(recH, total), pct(estP, total), pct(recP, total)));
        }
        out.println();
        out.println("Fin de la ejecución.");
    }

    static String pct(int a, int t) {
        return Math.round(100.0 * a / t) + "%";
    }

    static int producto(int[] a, int[] b) {
        int s = 0;
        for (int i = 0; i < N; i++) s += a[i] * b[i];
        return s;
    }

    /** % de consultas con ruido que terminan exactamente en el patrón de origen. */
    static long tasa(Hopfield red, List<int[]> patrones, double ruido, Random rnd) {
        int ok = 0, total = 100;
        for (int t = 0; t < total; t++) {
            int[] origen = patrones.get(t % patrones.size());
            int[] entrada = Imagen.conRuido(origen, ruido, rnd);
            if (iguales(red.consultar(entrada).salida, origen)) ok++;
        }
        return Math.round(100.0 * ok / total);
    }
}
