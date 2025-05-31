using Accord.MachineLearning.VectorMachines;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Statistics.Kernels;
using iTextSharp.text;
using iTextSharp.text.pdf;
using OpenCvSharp;
using OpenCvSharp.Extensions;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Reflection.Metadata;
using System.Reflection.PortableExecutable;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using Point = OpenCvSharp.Point;

namespace V8
{
    public class CameraProcessor
    {
        private readonly VideoCapture streamVideo;
        private bool isAnalizando = false;
        private bool isCapturing = false;
        private readonly Mat currentFrame = new();
        private readonly string username = "admin";
        private readonly string password = "TIAmerica24$";
        private readonly string ip;
        private readonly string endpoint = "cam/realmonitor?channel=1&subtype=0";
        private SupportVectorMachine<Gaussian> machine;
        private OpenCvSharp.Point[][] contours;
        private readonly string folderPath;
        private int imageCount = 0;
        private readonly Dictionary<string, string> registroErrores = new Dictionary<string, string>(); // Almacena los errores con su timestamp
        private DateTime lastErrorTime = DateTime.MinValue;
        private int imagesPerHour = 0;
        private readonly System.Timers.Timer pdfTimer;

        //Interfaz
        private DefectAnalysisForm analysisForm;
        public event Action<string, System.Drawing.Rectangle, Bitmap, double> DefectDetected;

        // Variables para evitar repeticiones de la misma zona marcada en rojo
        private Rect lastBoundingBox = default;
        private DateTime lastDrawTime = DateTime.MinValue;

        public void SetAnalysisForm(DefectAnalysisForm form)
        {
            analysisForm = form;
        }


        public CameraProcessor(string ipAddress, string folderPath)
        {
            ip = ipAddress;
            this.folderPath = folderPath;

            // Intentar inicializar el flujo de video, capturando cualquier error
            try
            {
                streamVideo = new VideoCapture($"rtsp://{username}:{password}@{ip}/{endpoint}");
                if (!streamVideo.IsOpened())
                {
                    throw new Exception("No se pudo conectar al flujo de video. Verifique la dirección IP y credenciales.");
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error al conectar con la cámara: {ex.Message}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }

            // Entrenar el modelo SVM
            EntrenamientoSVM();

            // Crear el folderPath si no existe
            if (!Directory.Exists(folderPath))
            {
                Directory.CreateDirectory(folderPath);
            }

            // Configurar el temporizador para generar el PDF cada hora
            pdfTimer = new System.Timers.Timer(TimeSpan.FromHours(1).TotalMilliseconds);
            pdfTimer.Elapsed += GeneratePdf;
            pdfTimer.AutoReset = true;
            pdfTimer.Enabled = true;
        }

        public async Task StartAnalysis(PictureBox pictureBox, int heightImage, int widthImage, PictureBox pictureBox2)
        {
            isAnalizando = true;

            while (isAnalizando)
            {
                await Task.Delay(10); // Esperar de manera asíncrona
                streamVideo.Read(currentFrame);

                if (currentFrame.Empty())
                {
                    MessageBox.Show("No se pudo capturar el fotograma. Deteniendo el análisis.", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    StopAnalysis();
                    break;
                }

                Analisis(currentFrame, pictureBox, heightImage, widthImage, pictureBox2);
            }
        }

        public void StopAnalysis()
        {
            isAnalizando = false;
        }



        // Modifica tu método Analisis para incluir notificaciones
        private void Analisis(Mat imagenAnalisis, PictureBox pictureBox, int heightImage, int widthImage, PictureBox pictureBox2)
        {
            try
            {
                using Mat filterImage = PreprocesarImagen(imagenAnalisis);
                using Mat thresholdImage = new();
                using Mat cannyEdges = new();

                Cv2.Threshold(filterImage, thresholdImage, 130, 170, ThresholdTypes.Binary);
                Cv2.Canny(thresholdImage, cannyEdges, 100, 200);

                Cv2.FindContours(cannyEdges, out contours, out _, RetrievalModes.List, ContourApproximationModes.ApproxSimple);
                List<Rect> positiveBoundingBoxes = ProcesarContornos(imagenAnalisis);
                List<Rect> filteredBoundingBoxes = FiltrarBoundingBoxes(positiveBoundingBoxes, 0.5);
                List<List<Rect>> groupedContours = GroupVerticallyAlignedContours(filteredBoundingBoxes, 0.1, 5);

                foreach (var group in groupedContours)
                {
                    if (group.Count >= 10)
                    {
                        Rect combinedBoundingBox = CombineBoundingBoxes(group);

                        if (ShouldDrawBoundingBox(combinedBoundingBox))
                        {
                            Cv2.Rectangle(imagenAnalisis, combinedBoundingBox, Scalar.Red, 4);
                            string label = "Positivo";

                            using Mat roi = new(imagenAnalisis, combinedBoundingBox);
                            Bitmap bmpRoi = BitmapConverter.ToBitmap(roi);

                            // NUEVO: Clasificar el tipo de defecto y notificar a la interfaz
                            string defectType = ClassifyDefect(roi, combinedBoundingBox);
                            double defectSize = CalculateDefectSize(combinedBoundingBox);

                            // Convertir OpenCV Rect a System.Drawing.Rectangle
                            var location = new System.Drawing.Rectangle(
                                combinedBoundingBox.X,
                                combinedBoundingBox.Y,
                                combinedBoundingBox.Width,
                                combinedBoundingBox.Height);

                            // Notificar a la interfaz
                            analysisForm?.AddDefect(defectType, location, (Bitmap)bmpRoi.Clone());
                            DefectDetected?.Invoke(defectType, location, (Bitmap)bmpRoi.Clone(), defectSize);

                            if (pictureBox2.InvokeRequired)
                            {
                                pictureBox2.Invoke(new Action(() =>
                                {
                                    pictureBox2.Image?.Dispose();
                                    pictureBox2.Image = bmpRoi;
                                }));
                            }
                            else
                            {
                                pictureBox2.Image?.Dispose();
                                pictureBox2.Image = bmpRoi;
                            }

                            Cv2.PutText(imagenAnalisis, $"{defectType} ({defectSize:F1}mm)",
                                       new OpenCvSharp.Point(combinedBoundingBox.X, combinedBoundingBox.Y - 10),
                                       HersheyFonts.HersheySimplex, 0.5, Scalar.Green, 2);

                            if (DateTime.Now.Subtract(lastErrorTime).TotalSeconds > 60)
                            {
                                string timestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
                                registroErrores[timestamp] = Path.Combine(folderPath, $"{imageCount}.jpg");
                                lastErrorTime = DateTime.Now;

                                string image_path = Path.Combine(folderPath, $"{imageCount}.jpg");
                                imageCount++;
                                imagesPerHour++;
                                Task.Run(() => Cv2.ImWrite(image_path, imagenAnalisis));
                            }
                        }
                    }
                }

                DisplayImage(imagenAnalisis, pictureBox, heightImage, widthImage);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error durante el análisis: {ex.Message}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        // NUEVO: Método para clasificar el tipo de defecto basado en las features
        private string ClassifyDefect(Mat roi, Rect boundingBox)
        {
            try
            {
                double[] features = ExtractEnhancedFeatures(roi);

                // Análisis basado en características específicas
                double lineCount = features.Length > 2 ? features[2] : 0;      // Número de líneas
                double circleCount = features.Length > 5 ? features[5] : 0;    // Número de círculos  
                double circularity = features.Length > 12 ? features[12] : 0;  // Circularidad
                double lbpVariance = features.Length > 7 ? features[7] : 0;    // Varianza LBP
                double aspectRatio = (double)boundingBox.Width / boundingBox.Height;

                // Lógica de clasificación mejorada
                if (circleCount > 0 && circularity > 0.6)
                    return "Pore";
                else if (lineCount > 3 && (aspectRatio > 3 || aspectRatio < 0.33))
                    return "Scratches";
                else if (lineCount > 2 && aspectRatio > 1.5 && aspectRatio < 4)
                    return "Crack";
                else if (lbpVariance > 20 && circularity < 0.4)
                    return "Ocode";  // Superficie irregular/golpeada
                else if (boundingBox.Width < 50 && boundingBox.Height < 50)
                    return "Spot";
                else
                    return "Seam";
            }
            catch
            {
                return "Crack"; // Default fallback
            }
        }

        // NUEVO: Método para calcular el tamaño del defecto en mm
        private double CalculateDefectSize(Rect boundingBox)
        {
            // Factor de conversión píxeles a mm (ajusta según tu cámara y distancia)
            double pixelsPerMm = 10.0; // Ejemplo: 10 píxeles = 1mm

            double widthMm = boundingBox.Width / pixelsPerMm;
            double heightMm = boundingBox.Height / pixelsPerMm;

            // Retornar la dimensión mayor
            return Math.Max(widthMm, heightMm);
        }

        // NUEVO: Método de inicio con interfaz integrada
        public async Task StartAnalysisWithInterface(DefectAnalysisForm form, PictureBox pictureBox,
                                                   int heightImage, int widthImage, PictureBox pictureBox2)
        {
            SetAnalysisForm(form);
            await StartAnalysis(pictureBox, heightImage, widthImage, pictureBox2);
        }


        private static Mat PreprocesarImagen(Mat imagenAnalisis)
        {
            Mat grayImage = new();
            Cv2.CvtColor(imagenAnalisis, grayImage, ColorConversionCodes.BGR2GRAY);

            // Máscara de interés
            Mat mask = Mat.Zeros(grayImage.Size(), MatType.CV_8UC1);
            OpenCvSharp.Point[] maskPoints = [
            new(865, 890), new(1988, 890), new(1920, 1280), new(844, 1280)
            ];
            Cv2.FillPoly(mask, new[] { maskPoints }, Scalar.White);

            Mat filteredImage = new();
            Cv2.BitwiseAnd(grayImage, grayImage, filteredImage, mask);
            Cv2.GaussianBlur(filteredImage, filteredImage, new OpenCvSharp.Size(5, 5), 0);

            return filteredImage;
        }

        private List<Rect> ProcesarContornos(Mat imagenAnalisis)
        {
            // Lista para almacenar las cajas delimitadoras de contornos positivos
            List<Rect> positiveBoundingBoxes = [];
            int contador = 0;

            // Objeto de bloqueo para acceso concurrente seguro a la lista y al contador
            object lockObject = new();

            // Paralelizar el procesamiento de contornos usando Parallel.ForEach
            Parallel.ForEach(contours, currentContour =>
            {
                Rect boundingBox = Cv2.BoundingRect(currentContour);
                int size = 90; // Tamaño ajustable según el tamaño del defecto

                // Definir la región de interés (ROI) centrada en el rectángulo delimitador
                Rect roi = new(boundingBox.X + boundingBox.Width / 2 - size / 2,
                               boundingBox.Y + boundingBox.Height / 2 - size / 2, size, size);
                roi &= new Rect(new OpenCvSharp.Point(0, 0), imagenAnalisis.Size());

                Mat contourImg = new(imagenAnalisis, roi);

                bool esDefecto = AnalizarContorno(contourImg);

                if (esDefecto)
                {
                    lock (lockObject)
                    {
                        positiveBoundingBoxes.Add(roi);
                    }
                }
            });

            return positiveBoundingBoxes;
        }

        private bool AnalizarContorno(Mat image)
        {
            try
            {
                var features = ExtractFeatures(image);
                return machine.Decide(features);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error al analizar el contorno: {ex.Message}");
                return false;
            }
            finally
            {
                image?.Dispose();
            }
        }

        private void GeneratePdf(object sender, System.Timers.ElapsedEventArgs e)
        {
            try
            {
                string pdfPath = Path.Combine(folderPath, $"Reporte_{DateTime.Now:yyyyMMdd_HHmmss}.pdf");
                using iTextSharp.text.Document pdfDocument = new();
                PdfWriter.GetInstance(pdfDocument, new FileStream(pdfPath, FileMode.Create));
                pdfDocument.Open();

                pdfDocument.Add(new Paragraph("Reporte de errores y capturas"));
                pdfDocument.Add(new Paragraph($"Total de imágenes capturadas en la última hora: {imagesPerHour}"));
                pdfDocument.Add(new Paragraph("Registro de errores:"));

                int imageLimit = 10; // Limitar imágenes en el PDF
                int count = 0;

                foreach (var registro in registroErrores)
                {
                    if (count >= imageLimit) break;

                    pdfDocument.Add(new Paragraph($"Hora: {registro.Key}, Imagen: {registro.Value}"));
                    var image = iTextSharp.text.Image.GetInstance(registro.Value);
                    image.ScalePercent(20f);
                    pdfDocument.Add(image);
                    count++;
                }

                pdfDocument.Close();
                registroErrores.Clear();
                imagesPerHour = 0;
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error al generar el PDF: {ex.Message}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }


        private static List<List<Rect>> GroupVerticallyAlignedContours(List<Rect> boundingBoxes, double tolerance, int minCount)
        {
            List<List<Rect>> groups = [];

            for (int i = 0; i < boundingBoxes.Count; i++)
            {
                var rect1 = boundingBoxes[i];
                var group = new List<Rect> { rect1 };

                for (int j = i + 1; j < boundingBoxes.Count; j++)
                {
                    var rect2 = boundingBoxes[j];

                    if (Math.Abs(rect1.X - rect2.X) <= rect1.Width * tolerance)
                    {
                        group.Add(rect2);
                    }
                }

                if (group.Count >= minCount)
                {
                    groups.Add(group);
                }
            }

            return groups;
        }

        private static Rect CombineBoundingBoxes(List<Rect> boundingBoxes)
        {
            int x = boundingBoxes.Min(r => r.X);
            int y = boundingBoxes.Min(r => r.Y);
            int width = boundingBoxes.Max(r => r.Right) - x;
            int height = boundingBoxes.Max(r => r.Bottom) - y;

            return new Rect(x, y, width, height);
        }

        private static double[] ExtractFeatures(Mat image)
        {
            Cv2.MeanStdDev(image, out Scalar mean, out Scalar stddev);

            var features = new double[2];
            features[0] = mean.Val0;   // Media
            features[1] = stddev.Val0; // Desviación estándar

            return features;
        }

        private static double[] ExtractEnhancedFeatures(Mat image)
        {
            var features = new List<double>();

            // 1. Features básicos (los que ya tienes)
            Cv2.MeanStdDev(image, out Scalar mean, out Scalar stddev);
            features.Add(mean.Val0);
            features.Add(stddev.Val0);

            // 2. Features para RAYONES (detecta líneas)
            Mat edges = new();
            Cv2.Canny(image, edges, 50, 150);

            // Transformada de Hough para líneas
            LineSegmentPoint[] lines = Cv2.HoughLinesP(edges, 1, Math.PI / 180, 50, 30, 10);
            features.Add(lines.Length); // Número de líneas detectadas

            // Orientación predominante de líneas (rayones suelen ser horizontales/verticales)
            if (lines.Length > 0)
            {
                var angles = lines.Select(line =>
                    Math.Atan2(line.P2.Y - line.P1.Y, line.P2.X - line.P1.X) * 180 / Math.PI).ToArray();
                features.Add(angles.Average()); // Ángulo promedio
                features.Add(angles.Max() - angles.Min()); // Variación angular
            }
            else
            {
                features.Add(0);
                features.Add(0);
            }

            // 3. Features para AGUJEROS (detecta círculos/elipses)
            CircleSegment[] circles = Cv2.HoughCircles(image, HoughModes.Gradient, 1, 20, 50, 30, 5, 50);
            features.Add(circles.Length); // Número de círculos

            // Área total de círculos detectados
            double totalCircleArea = circles.Sum(c => Math.PI * c.Radius * c.Radius);
            features.Add(totalCircleArea);

            // 4. Features para SUPERFICIES GOLPEADAS (detecta irregularidades)
            // Análisis de textura con LBP (Local Binary Pattern)
            var lbpVariance = CalculateLBPVariance(image);
            features.Add(lbpVariance);

            // Gradiente morfológico (detecta bordes irregulares)
            Mat morphGrad = new();
            var kernel = Cv2.GetStructuringElement(MorphShapes.Ellipse, new OpenCvSharp.Size(3, 3));
            Cv2.MorphologyEx(image, morphGrad, MorphTypes.Gradient, kernel);
            Cv2.MeanStdDev(morphGrad, out Scalar morphMean, out Scalar morphStd);
            features.Add(morphMean.Val0);
            features.Add(morphStd.Val0);

            // 5. Features de contraste local (detecta cambios abruptos)
            Mat laplacian = new();
            Cv2.Laplacian(image, laplacian, MatType.CV_64F);
            Cv2.MeanStdDev(laplacian, out Scalar lapMean, out Scalar lapStd);
            features.Add(Math.Abs(lapMean.Val0));
            features.Add(lapStd.Val0);

            // 6. Ratio de área vs perímetro (agujeros tienen alta circularidad)
            Cv2.FindContours(edges, out Point[][] contours, out _, RetrievalModes.External, ContourApproximationModes.ApproxSimple);
            if (contours.Length > 0)
            {
                var largestContour = contours.OrderByDescending(c => Cv2.ContourArea(c)).First();
                double area = Cv2.ContourArea(largestContour);
                double perimeter = Cv2.ArcLength(largestContour, true);
                double circularity = perimeter > 0 ? (4 * Math.PI * area) / (perimeter * perimeter) : 0;
                features.Add(circularity);
            }
            else
            {
                features.Add(0);
            }

            // Cleanup
            edges.Dispose();
            morphGrad.Dispose();
            kernel.Dispose();
            laplacian.Dispose();

            return features.ToArray();
        }

        private static double CalculateLBPVariance(Mat image)
        {
            var lbpValues = new List<int>();

            for (int y = 1; y < image.Height - 1; y++)
            {
                for (int x = 1; x < image.Width - 1; x++)
                {
                    byte centerPixel = image.At<byte>(y, x);
                    int lbpValue = 0;

                    // Comparar con 8 vecinos en sentido horario
                    if (image.At<byte>(y - 1, x - 1) >= centerPixel) lbpValue |= 1;
                    if (image.At<byte>(y - 1, x) >= centerPixel) lbpValue |= 2;
                    if (image.At<byte>(y - 1, x + 1) >= centerPixel) lbpValue |= 4;
                    if (image.At<byte>(y, x + 1) >= centerPixel) lbpValue |= 8;
                    if (image.At<byte>(y + 1, x + 1) >= centerPixel) lbpValue |= 16;
                    if (image.At<byte>(y + 1, x) >= centerPixel) lbpValue |= 32;
                    if (image.At<byte>(y + 1, x - 1) >= centerPixel) lbpValue |= 64;
                    if (image.At<byte>(y, x - 1) >= centerPixel) lbpValue |= 128;

                    lbpValues.Add(lbpValue);
                }
            }

            // Calcular varianza de valores LBP
            if (lbpValues.Count == 0) return 0;

            double mean = lbpValues.Average();
            double variance = lbpValues.Sum(x => Math.Pow(x - mean, 2)) / lbpValues.Count;
            return variance;
        }

        /*private void EntrenamientoSVM()
        {
            machine = GetMachine();

            // Aquí se deberían cargar los datos de entrenamiento desde un dataset adecuado.
            double[][] inputs = [[1, 1], [2, 2], [3, 3]];
            int[] outputs = [1, -1, 1];

            var teacher = new SequentialMinimalOptimization<Gaussian>()
            {
                Complexity = 100
            };

            machine = teacher.Learn(inputs, outputs);
        }*/

        private static SupportVectorMachine<Gaussian> GetMachine()
        {
            var gaussianKernel = new Gaussian(0.1);
            return new SupportVectorMachine<Gaussian>(inputs: 2, kernel: gaussianKernel);
        }

        private static void DisplayImage(Mat image, PictureBox pictureBox, int heightImage, int widthImage)
        {
            if (pictureBox.InvokeRequired)
            {
                pictureBox.Invoke(new MethodInvoker(delegate { DisplayImage(image, pictureBox, heightImage, widthImage); }));
            }
            else
            {
                Bitmap bitmap = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(image);
                pictureBox.Image?.Dispose();
                pictureBox.Image = new Bitmap(bitmap, widthImage, heightImage);
            }
        }

        private void EntrenamientoSVM()
        {
            // Crear datasets específicos para cada tipo de defecto
            var trainingData = new List<(double[] features, int label, string defectType)>();

            // Cargar imágenes de entrenamiento (deberías tener carpetas separadas)
            string basePath = Path.Combine(Application.StartupPath, "TrainingData");

            // Clase 0: Superficie normal
            LoadTrainingImages(Path.Combine(basePath, "Normal"), 0, "Normal", trainingData);

            // Clase 1: Rayones
            LoadTrainingImages(Path.Combine(basePath, "Rayones"), 1, "Rayon", trainingData);

            // Clase 2: Agujeros
            LoadTrainingImages(Path.Combine(basePath, "Agujeros"), 2, "Agujero", trainingData);

            // Clase 3: Superficies golpeadas
            LoadTrainingImages(Path.Combine(basePath, "Golpeadas"), 3, "Golpe", trainingData);

            if (trainingData.Count == 0)
            {
                // Fallback con datos sintéticos mejorados
                CreateSyntheticTrainingData(trainingData);
            }

            // Preparar datos para SVM
            double[][] inputs = trainingData.Select(x => x.features).ToArray();
            int[] outputs = trainingData.Select(x => x.label).ToArray();

            // Normalizar features (muy importante para SVM)
            inputs = NormalizeFeatures(inputs);

            // Configurar SVM con kernel RBF optimizado
            var gaussianKernel = new Gaussian(0.5); // Gamma ajustado para tus features
            machine = new SupportVectorMachine<Gaussian>(inputs: inputs[0].Length, kernel: gaussianKernel);

            // Configurar el algoritmo de aprendizaje
            var teacher = new SequentialMinimalOptimization<Gaussian>()
            {
                Complexity = 10.0, // C parameter - ajusta según tu dataset
                Tolerance = 1e-3,
                UseComplexityHeuristic = true,
                UseKernelEstimation = true
            };

            try
            {
                machine = teacher.Learn(inputs, outputs);

                // Validación cruzada simple
                double accuracy = ValidateModel(inputs, outputs);
                Console.WriteLine($"Precisión del modelo: {accuracy:P2}");

                if (accuracy < 0.8) // Si la precisión es baja
                {
                    MessageBox.Show($"⚠️ Precisión del modelo baja ({accuracy:P1}). Considera agregar más datos de entrenamiento.",
                                  "Advertencia", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error en entrenamiento SVM: {ex.Message}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                // Fallback a modelo simple
                CreateSimpleFallbackModel();
            }
        }

        private void LoadTrainingImages(string folderPath, int label, string defectType,
                                       List<(double[] features, int label, string defectType)> trainingData)
        {
            if (!Directory.Exists(folderPath)) return;

            var imageFiles = Directory.GetFiles(folderPath, "*.jpg")
                                    .Concat(Directory.GetFiles(folderPath, "*.png"))
                                    .Concat(Directory.GetFiles(folderPath, "*.bmp"));

            foreach (string imagePath in imageFiles)
            {
                try
                {
                    using Mat image = Cv2.ImRead(imagePath, ImreadModes.Grayscale);
                    if (!image.Empty())
                    {
                        // Redimensionar a tamaño estándar para consistency
                        using Mat resized = new();
                        Cv2.Resize(image, resized, new OpenCvSharp.Size(90, 90));

                        double[] features = ExtractEnhancedFeatures(resized);
                        trainingData.Add((features, label, defectType));
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error cargando imagen {imagePath}: {ex.Message}");
                }
            }
        }

        private static double[][] NormalizeFeatures(double[][] features)
        {
            if (features.Length == 0) return features;

            int featureCount = features[0].Length;
            double[][] normalized = new double[features.Length][];

            // Calcular min/max para cada feature
            double[] mins = new double[featureCount];
            double[] maxs = new double[featureCount];

            for (int f = 0; f < featureCount; f++)
            {
                mins[f] = features.Min(x => x[f]);
                maxs[f] = features.Max(x => x[f]);
            }

            // Normalizar a rango [0, 1]
            for (int i = 0; i < features.Length; i++)
            {
                normalized[i] = new double[featureCount];
                for (int f = 0; f < featureCount; f++)
                {
                    double range = maxs[f] - mins[f];
                    normalized[i][f] = range > 0 ? (features[i][f] - mins[f]) / range : 0;
                }
            }

            return normalized;
        }

        private double ValidateModel(double[][] inputs, int[] outputs)
        {
            int correct = 0;
            for (int i = 0; i < inputs.Length; i++)
            {
                bool prediction = machine.Decide(inputs[i]);
                bool actual = outputs[i] > 0; // Convert multi-class to binary for validation
                if (prediction == actual) correct++;
            }
            return (double)correct / inputs.Length;
        }

        private void CreateSyntheticTrainingData(List<(double[] features, int label, string defectType)> trainingData)
        {
            // Datos sintéticos más realistas basados en características típicas

            // Superficie normal: baja varianza, sin líneas, sin círculos
            for (int i = 0; i < 20; i++)
            {
                trainingData.Add((new double[] {
            Random.Shared.NextDouble() * 10 + 100,  // mean
            Random.Shared.NextDouble() * 5 + 2,     // std
            0, 0, 0,  // sin líneas
            0, 0,     // sin círculos
            Random.Shared.NextDouble() * 10 + 5,    // lbp variance baja
            Random.Shared.NextDouble() * 10 + 10,   // morph mean
            Random.Shared.NextDouble() * 5 + 2,     // morph std
            Random.Shared.NextDouble() * 5 + 1,     // laplacian mean
            Random.Shared.NextDouble() * 3 + 1,     // laplacian std
            Random.Shared.NextDouble() * 0.2 + 0.1  // circularidad baja
        }, 0, "Normal"));
            }

            // Rayones: muchas líneas, alta varianza direccional
            for (int i = 0; i < 15; i++)
            {
                trainingData.Add((new double[] {
            Random.Shared.NextDouble() * 20 + 80,   // mean variable
            Random.Shared.NextDouble() * 10 + 8,    // std alta
            Random.Shared.NextDouble() * 10 + 5,    // muchas líneas
            Random.Shared.NextDouble() * 180,       // ángulo variable
            Random.Shared.NextDouble() * 45 + 10,   // variación angular
            0, 0,     // sin círculos
            Random.Shared.NextDouble() * 20 + 15,   // lbp variance alta
            Random.Shared.NextDouble() * 15 + 15,   // morph mean alta
            Random.Shared.NextDouble() * 8 + 5,     // morph std alta
            Random.Shared.NextDouble() * 10 + 5,    // laplacian mean alta
            Random.Shared.NextDouble() * 6 + 3,     // laplacian std alta
            Random.Shared.NextDouble() * 0.3 + 0.1  // circularidad baja
        }, 1, "Rayon"));
            }

            // Agujeros: círculos detectados, alta circularidad
            for (int i = 0; i < 15; i++)
            {
                trainingData.Add((new double[] {
            Random.Shared.NextDouble() * 30 + 60,   // mean muy variable
            Random.Shared.NextDouble() * 15 + 10,   // std muy alta
            Random.Shared.NextDouble() * 3,         // pocas líneas
            Random.Shared.NextDouble() * 180,       // ángulo irrelevante
            Random.Shared.NextDouble() * 20,        // variación baja
            Random.Shared.NextDouble() * 5 + 1,     // círculos detectados
            Random.Shared.NextDouble() * 500 + 100, // área círculos
            Random.Shared.NextDouble() * 25 + 20,   // lbp variance alta
            Random.Shared.NextDouble() * 20 + 20,   // morph mean muy alta
            Random.Shared.NextDouble() * 10 + 8,    // morph std muy alta
            Random.Shared.NextDouble() * 12 + 8,    // laplacian mean muy alta
            Random.Shared.NextDouble() * 8 + 5,     // laplacian std muy alta
            Random.Shared.NextDouble() * 0.6 + 0.4  // circularidad alta
        }, 2, "Agujero"));
            }
        }

        private void CreateSimpleFallbackModel()
        {
            // Modelo extremadamente simple como fallback
            var gaussianKernel = new Gaussian(0.1);
            machine = new SupportVectorMachine<Gaussian>(inputs: 2, kernel: gaussianKernel);

            double[][] simpleInputs = { new double[] { 1, 1 }, new double[] { 0, 0 } };
            int[] simpleOutputs = { 1, 0 };

            var teacher = new SequentialMinimalOptimization<Gaussian>() { Complexity = 1 };
            machine = teacher.Learn(simpleInputs, simpleOutputs);
        }

        public void StopReproducirVideo()
        {
            isCapturing = false;
        }

        public void ReproducirVideo(PictureBox pictureBox)
        {
            isCapturing = true;

            Task.Run(() =>
            {
                while (isCapturing)
                {
                    streamVideo.Read(currentFrame);

                    if (currentFrame.Empty())
                    {
                        MessageBox.Show("No se pudo capturar el video.", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                        break;
                    }

                    pictureBox.Invoke(new Action(() =>
                    {
                        DisplayImage(currentFrame, pictureBox, 640, 480);
                    }));

                    Thread.Sleep(30);
                }
            });
        }

        // Métodos nuevos para evitar repeticiones:
        private bool ShouldDrawBoundingBox(Rect currentBox)
        {
            double overlapThreshold = 0.5;
            int timeThresholdSeconds = 120;

            TimeSpan timeSinceLastDraw = DateTime.Now - lastDrawTime;

            if (timeSinceLastDraw.TotalSeconds < timeThresholdSeconds && lastBoundingBox != default)
            {
                double overlap = ComputeOverlap(lastBoundingBox, currentBox);
                if (overlap > overlapThreshold)
                {
                    // La zona es esencialmente la misma y se ha dibujado muy recientemente
                    return false;
                }
            }

            // Actualizar registro
            lastDrawTime = DateTime.Now;
            lastBoundingBox = currentBox;
            return true;
        }

        private static double ComputeOverlap(Rect a, Rect b)
        {
            int x1 = Math.Max(a.Left, b.Left);
            int y1 = Math.Max(a.Top, b.Top);
            int x2 = Math.Min(a.Right, b.Right);
            int y2 = Math.Min(a.Bottom, b.Bottom);

            int intersection = Math.Max(0, x2 - x1) * Math.Max(0, y2 - y1);
            int unionArea = a.Width * a.Height + b.Width * b.Height - intersection;

            if (unionArea > 0)
                return (double)intersection / unionArea;
            else
                return 0;
        }

        private static List<Rect> FiltrarBoundingBoxes(List<Rect> boxes, double overlapThreshold)
        {
            boxes.Sort((a, b) => a.Y.CompareTo(b.Y)); // Ordenar por posición vertical
            List<Rect> filtered = [];

            foreach (var box in boxes)
            {
                if (filtered.Any(f => ComputeOverlap(f, box) > overlapThreshold))
                    continue; // Ignorar si hay mucha superposición
                filtered.Add(box);
            }

            return filtered;
        }
        /* private static void DibujarBoundingBoxes(Mat imagen, List<List<Rect>> grupos)
         {
             Random rand = new();
             foreach (var grupo in grupos)
             {
                 Scalar color = new(rand.Next(256), rand.Next(256), rand.Next(256));
                 foreach (var box in grupo)
                 {
                     Cv2.Rectangle(imagen, box, color, 2);
                 }
             }
         }*/


    }

}