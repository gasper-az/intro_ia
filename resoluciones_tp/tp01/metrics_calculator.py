import statistics
import tracemalloc
import time

class MetricsCalculator:
    """
    Permite obtener métricas sobre las ejecuciones de una función dada
    """

    def __init__(self, funcion_a_ejecutar, cantidad_de_ejecuciones = 10):
        """
        Args:
            funcion_a_ejecutar (lambda): Función sobre la cual se van a obtener ciertas métricas.
            cantidad_de_ejecuciones (int): Cantidad de ejecuciones que se realizarán sobre `funcion_a_ejecutar`.
        """
        self.funcion = funcion_a_ejecutar
        self.cantidad_de_ejecuciones = cantidad_de_ejecuciones
    
    def calcular_memoria(self):
        """
        Permite obtener métricas relacionadas al consumo de memoria al ejecutar una función dada una
        cantidad dada de veces.

        Returns:
            Dictionary. Diccionario con ciertas métricas referidas al consumo de memoria, incluyendo el promedio,
            desvío estándar, máximo y mínimo.
        """
        resultados_calc_memoria = []

        for i in range(self.cantidad_de_ejecuciones):
            tracemalloc.start()

            solution, metrics = self.funcion()

            # Para medir memoria consumida usamos el pico de memoria
            _, memory_peak = tracemalloc.get_traced_memory()
            memory_peak /= 1024*1024
            tracemalloc.stop()

            data = {
                "nro_ejecucion": i,
                "solution": solution,
                "metrics": metrics,
                "memory_peak_mb": memory_peak
            }

            # Insertamos al inicio del array
            resultados_calc_memoria.insert(0, data)
        
        # obtenemos solo los memory peaks
        memory_peaks = [x["memory_peak_mb"] for x in resultados_calc_memoria]
        promedio = statistics.mean(memory_peaks)
        desvio = statistics.stdev(memory_peaks)

        return {
            "resultados": resultados_calc_memoria,
            "promedio": promedio,
            "desvio": desvio,
            "maximo": max(memory_peaks),
            "minimo": min(memory_peaks)
        }
    
    def calcular_tiempo(self):
        """
        Permite obtener métricas relacionadas al tiempo de ejecución de una función dada una
        cantidad dada de veces.

        Returns:
            Dictionary. Diccionario con ciertas métricas referidas a tiempos de ejecución, incluyendo el promedio,
            desvío estándar, máximo y mínimo.
        """
        resultados_calc_tiempo = []

        for i in range(self.cantidad_de_ejecuciones):
            start_time = time.perf_counter()
            solution, metrics = self.funcion()
            end_time = time.perf_counter()

            data = {
                "nro_ejecucion": i,
                "solution": solution,
                "metrics": metrics,
                "time_seconds": (end_time-start_time)
            }

            # Insertamos al inicio del array
            resultados_calc_tiempo.insert(0, data)
        
        time_spent = [x["time_seconds"] for x in resultados_calc_tiempo]
        promedio = statistics.mean(time_spent)
        desvio = statistics.stdev(time_spent)

        return {
            "resultados": resultados_calc_tiempo,
            "promedio": promedio,
            "desvio": desvio,
            "maximo": max(time_spent),
            "minimo": min(time_spent)
        }
    
    def ejecutar_y_analizar_trayectos(self):
        """
        Permite ejecutar una función una cantidad de veces dada.

        Returns:
            Dictionary. Diccionario que contiene todos los resultados de la ejecución de una función dada,
            incluyendo el promedio, desvío estándar, máximo, y mínimo del `solution_depth` de cada ejecución.
        """

        resultados = []

        for i in range(self.cantidad_de_ejecuciones):
            solution, metrics = self.funcion()
            
            data = {
                "nro_ejecucion": i,
                "solution": solution,
                "metrics": metrics,
                "solution_depth": solution.depth
            }

            resultados.insert(0, data)

        depth = [x["solution_depth"] for x in resultados]
        promedio = statistics.mean(depth)
        desvio = statistics.stdev(depth)

        return {
            "resultados": resultados,
            "promedio": promedio,
            "desvio": desvio,
            "maximo": max(depth),
            "minimo": min(depth)
        }