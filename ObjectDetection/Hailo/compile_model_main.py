import os
import sys
import logging
import psutil
import time
import datetime
from threading import Thread
from pathlib import Path
from IPython.display import clear_output, display, Image
import subprocess
import queue
import shutil
import glob

# Configuración de directorios y logs
class Config:
    def __init__(self):
        self.log_dir = Path('log')
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.plots_dir = self.log_dir / f'plots_{self.timestamp}'
        self.log_file = self.log_dir / f'compilation_log_{self.timestamp}.txt'

        # Crear directorios necesarios
        self.log_dir.mkdir(exist_ok=True)
        self.plots_dir.mkdir(exist_ok=True)

        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )

class ResourceMonitor:
    def __init__(self):
        self.config = Config()
        self.compile_output = []

    @staticmethod
    def format_bytes(bytes_value):
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_value < 1024:
                return f"{bytes_value:.2f}{unit}"
            bytes_value /= 1024
        return f"{bytes_value:.2f}TB"

    def log_output(self, line):
        with open(self.config.log_file, 'a', encoding='utf-8') as f:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"[{timestamp}] {line}\n")

    def create_progress_bar(self, percentage, length=25):
        filled = int(length * percentage / 100)
        return f"{'█' * filled}{'░' * (length - filled)}"

    def print_status(self, cpu, mem, disk):
        clear_output(wait=True)
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Cabecera
        print(f"\n{'═' * 50}")
        print(f"  MONITOR DE RECURSOS - {current_time}")
        print(f"{'═' * 50}\n")

        # Estado de Compilación
        if self.compile_output:
            print("ESTADO DE COMPILACIÓN (últimas 5 líneas):")
            print(f"{'─' * 50}")
            for line in self.compile_output[-5:]:
                print(f"  {line}")
            print(f"{'─' * 50}\n")

        # Uso de Recursos
        print("USO DE RECURSOS:")
        print(f"CPU:  {self.create_progress_bar(cpu)} {cpu:>5.1f}%")
        print(f"RAM:  {self.create_progress_bar(mem.percent)} {mem.percent:>5.1f}%")
        print(f"DISK: {self.create_progress_bar(disk.percent)} {disk.percent:>5.1f}%")

        # Detalles de Memoria
        print("\nDETALLES DE MEMORIA:")
        print(f"  Total: {self.format_bytes(mem.total):>10}")
        print(f"  Usado: {self.format_bytes(mem.used):>10}")
        print(f"  Libre: {self.format_bytes(mem.available):>10}")

    def monitor_resources(self):
        while True:
            try:
                cpu = psutil.cpu_percent(interval=1)
                mem = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                self.print_status(cpu, mem, disk)
            except Exception as e:
                logging.error(f"Error en monitoreo: {str(e)}")
            time.sleep(2)

    def read_output(self, pipe, queue):
        try:
            for line in pipe:
                line = line.strip()
                if line:  # Solo procesar líneas no vacías
                    queue.put(line)
                    self.log_output(line)
                    self.compile_output.append(line)
                    if len(self.compile_output) > 100:  # Mantener solo últimas 100 líneas
                        self.compile_output.pop(0)
        finally:
            if hasattr(pipe, 'close'):
                pipe.close()

    def run(self):
        try:
            print("\n=== INICIANDO MONITOREO DE SISTEMA ===")
            print(f"Logs: {self.config.log_file}")
            print(f"Plots: {self.config.plots_dir}\n")

            output_queue = queue.Queue()

            # Iniciar thread de monitoreo
            monitor_thread = Thread(target=self.monitor_resources, daemon=True)
            monitor_thread.start()

            # Configurar y ejecutar proceso de compilación
            env_path = Path('my_env/bin/python')
            compile_script = Path('compile_model.py')

            if not env_path.exists():
                raise FileNotFoundError(f"Entorno virtual no encontrado en {env_path}")
            if not compile_script.exists():
                raise FileNotFoundError(f"Script no encontrado en {compile_script}")

            process = subprocess.Popen(
                [str(env_path), str(compile_script)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )

            # Iniciar threads para stdout y stderr
            for pipe in [process.stdout, process.stderr]:
                Thread(target=self.read_output,
                      args=(pipe, output_queue),
                      daemon=True).start()

            # Esperar a que termine el proceso
            return_code = process.wait()

            if return_code != 0:
                print(f"\nError en la compilación (código {return_code})")
            else:
                print("\nCompilación completada exitosamente")

        except KeyboardInterrupt:
            print("\nDetención manual del monitoreo")
            if 'process' in locals():
                process.terminate()
        except Exception as e:
            logging.error(f"Error en la ejecución: {str(e)}")
            raise
        finally:
            print(f"\nLogs guardados en: {self.config.log_file}")
            print(f"Gráficos guardados en: {self.config.plots_dir}")
            sys.exit(0)

if __name__ == "__main__":
    monitor = ResourceMonitor()
    monitor.run()