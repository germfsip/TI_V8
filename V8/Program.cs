using System;
using System.Threading.Tasks;

namespace V8
{
    public class CameraProcessor
    {
        private bool isAnalyzing = false;
        private readonly string ipAddress;
        private readonly string folderPath;

        public bool IsAnalyzing => isAnalyzing;

        // Constructor que coincide con tu Program.cs
        public CameraProcessor(string ipAddress, string folderPath)
        {
            this.ipAddress = ipAddress;
            this.folderPath = folderPath;

            // Crear carpeta si no existe
            if (!System.IO.Directory.Exists(folderPath))
            {
                System.IO.Directory.CreateDirectory(folderPath);
            }
        }

        public async Task StartAnalysis()
        {
            isAnalyzing = true;

            // Por ahora solo simular - aquí iría tu lógica real
            await Task.Run(() =>
            {
                SimulateDefectDetection();
            });
        }

        public void StopAnalysis()
        {
            isAnalyzing = false;
        }

        // Evento para notificar cuando se detecten defectos
        public event Action<string, System.Drawing.Rectangle, System.Drawing.Bitmap> DefectDetected;

        protected virtual void OnDefectDetected(string defectType, System.Drawing.Rectangle location, System.Drawing.Bitmap image)
        {
            DefectDetected?.Invoke(defectType, location, image);
        }

        // Simulación simple para testing
        private async void SimulateDefectDetection()
        {
            var random = new Random();
            string[] defectTypes = { "Scratches", "Crack", "Spot", "Pore", "Seam", "Ocode" };

            while (isAnalyzing)
            {
                await Task.Delay(5000); // Cada 5 segundos

                if (!isAnalyzing) break;

                // Simular detección
                string defectType = defectTypes[random.Next(defectTypes.Length)];
                var location = new System.Drawing.Rectangle(
                    random.Next(0, 1920),
                    random.Next(0, 1080),
                    50, 50);

                // Crear imagen simple
                var bitmap = new System.Drawing.Bitmap(100, 100);
                using (var g = System.Drawing.Graphics.FromImage(bitmap))
                {
                    g.Clear(System.Drawing.Color.Red);
                    g.DrawString(defectType, new System.Drawing.Font("Arial", 8),
                                System.Drawing.Brushes.White, 5, 5);
                }

                OnDefectDetected(defectType, location, bitmap);
            }
        }
    }
}