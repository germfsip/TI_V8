using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Windows.Forms;
using OpenCvSharp;
using OpenCvSharp.Extensions;

namespace V8
{
    public partial class DefectAnalysisForm : Form
    {
        private readonly CameraProcessor cameraProcessor;
        private readonly System.Windows.Forms.Timer refreshTimer;
        private readonly Dictionary<string, int> defectCounts;
        private readonly List<DetectedDefect> recentDefects;
        private readonly Random colorGenerator = new();

        // Controls
        private Panel headerPanel;
        private Label titleLabel;
        private TableLayoutPanel defectCountsPanel;
        private PictureBox defectLocationPanel;
        private PictureBox errorPreviewPanel;
        private Panel controlPanel;
        private Button startC1Button, startC2Button, stopButton;
        private Label statusLabel;
        private ProgressBar analysisProgress;

        // Defect tracking
        private readonly List<DefectCountCard> defectCards = new();

        public DefectAnalysisForm(CameraProcessor processor)
        {
            cameraProcessor = processor;
            defectCounts = new Dictionary<string, int>
            {
                ["Scratches"] = 0,
                ["Crack"] = 0,
                ["Spot"] = 0,
                ["Pore"] = 0,
                ["Seam"] = 0,
                ["Ocode"] = 0
            };
            recentDefects = new List<DetectedDefect>();

            InitializeComponent();
            SetupRefreshTimer();

            // Conectar eventos del procesador de cámara
            if (cameraProcessor != null)
            {
                cameraProcessor.DefectDetected += OnDefectDetected;
            }
        }

        private void OnDefectDetected(string defectType, System.Drawing.Rectangle location, System.Drawing.Bitmap image)
        {
            if (this.InvokeRequired)
            {
                this.Invoke(new Action(() => AddDefect(defectType, location, image)));
            }
            else
            {
                AddDefect(defectType, location, image);
            }
        }

        private void InitializeComponent()
        {
            // Form configuration
            this.Size = new System.Drawing.Size(1400, 900);
            this.Text = "ANÁLISIS DE DEFECTOS - SUPERFICIES METÁLICAS";
            this.BackColor = Color.FromArgb(45, 45, 45);
            this.FormBorderStyle = FormBorderStyle.FixedDialog;
            this.MaximizeBox = false;
            this.StartPosition = FormStartPosition.CenterScreen;

            CreateHeaderPanel();
            CreateDefectCountsPanel();
            CreateDefectLocationPanel();
            CreateErrorPreviewPanel();
            CreateControlPanel();

            // Layout
            this.Controls.Add(headerPanel);
            this.Controls.Add(defectCountsPanel);
            this.Controls.Add(defectLocationPanel);
            this.Controls.Add(errorPreviewPanel);
            this.Controls.Add(controlPanel);
        }

        private void CreateHeaderPanel()
        {
            headerPanel = new Panel
            {
                Size = new System.Drawing.Size(1380, 80),
                Location = new System.Drawing.Point(10, 10),
                BackColor = Color.FromArgb(190, 50, 60)
            };

            titleLabel = new Label
            {
                Text = "ANÁLISIS 1 CÁMARA 📷",
                Font = new Font("Segoe UI", 24, FontStyle.Bold),
                ForeColor = Color.White,
                Size = new System.Drawing.Size(600, 60),
                Location = new System.Drawing.Point(20, 10),
                TextAlign = ContentAlignment.MiddleLeft
            };

            // Status indicator
            statusLabel = new Label
            {
                Text = "● DETENIDO",
                Font = new Font("Segoe UI", 12, FontStyle.Bold),
                ForeColor = Color.Orange,
                Size = new System.Drawing.Size(200, 30),
                Location = new System.Drawing.Point(1150, 25),
                TextAlign = ContentAlignment.MiddleRight
            };

            headerPanel.Controls.Add(titleLabel);
            headerPanel.Controls.Add(statusLabel);
        }

        private void CreateDefectCountsPanel()
        {
            defectCountsPanel = new TableLayoutPanel
            {
                Size = new System.Drawing.Size(900, 300),
                Location = new System.Drawing.Point(10, 100),
                BackColor = Color.FromArgb(55, 55, 55),
                CellBorderStyle = TableLayoutPanelCellBorderStyle.Single
            };

            defectCountsPanel.ColumnCount = 3;
            defectCountsPanel.RowCount = 2;

            // Configure columns and rows
            for (int i = 0; i < 3; i++)
                defectCountsPanel.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 33.33f));

            for (int i = 0; i < 2; i++)
                defectCountsPanel.RowStyles.Add(new RowStyle(SizeType.Percent, 50f));

            // Create defect cards
            string[] defectTypes = { "Scratches", "Crack", "Spot", "Pore", "Seam", "Ocode" };
            Color[] cardColors = {
                Color.FromArgb(100, 150, 255), // Blue
                Color.FromArgb(255, 100, 100), // Red
                Color.FromArgb(255, 200, 100), // Orange
                Color.FromArgb(150, 255, 150), // Green
                Color.FromArgb(255, 150, 255), // Pink
                Color.FromArgb(150, 255, 255)  // Cyan
            };

            for (int i = 0; i < defectTypes.Length; i++)
            {
                var card = new DefectCountCard(defectTypes[i], cardColors[i]);
                defectCards.Add(card);
                defectCountsPanel.Controls.Add(card, i % 3, i / 3);
            }
        }

        private void CreateDefectLocationPanel()
        {
            var locationContainer = new Panel
            {
                Size = new System.Drawing.Size(460, 300),
                Location = new System.Drawing.Point(920, 100),
                BackColor = Color.FromArgb(55, 55, 55),
                BorderStyle = BorderStyle.FixedSingle
            };

            var locationTitle = new Label
            {
                Text = "MAPA DE DEFECTOS",
                Font = new Font("Segoe UI", 12, FontStyle.Bold),
                ForeColor = Color.White,
                Size = new System.Drawing.Size(440, 30),
                Location = new System.Drawing.Point(10, 5),
                TextAlign = ContentAlignment.MiddleCenter
            };

            defectLocationPanel = new PictureBox
            {
                Size = new System.Drawing.Size(440, 260),
                Location = new System.Drawing.Point(10, 35),
                BackColor = Color.FromArgb(70, 70, 70),
                BorderStyle = BorderStyle.FixedSingle,
                SizeMode = PictureBoxSizeMode.Zoom
            };

            locationContainer.Controls.Add(locationTitle);
            locationContainer.Controls.Add(defectLocationPanel);
            this.Controls.Add(locationContainer);
        }

        private void CreateErrorPreviewPanel()
        {
            var previewContainer = new Panel
            {
                Size = new System.Drawing.Size(1370, 200),
                Location = new System.Drawing.Point(10, 420),
                BackColor = Color.FromArgb(35, 35, 35),
                BorderStyle = BorderStyle.FixedSingle
            };

            var previewTitle = new Label
            {
                Text = "MUESTRA ERROR",
                Font = new Font("Segoe UI", 14, FontStyle.Bold),
                ForeColor = Color.White,
                Size = new System.Drawing.Size(200, 30),
                Location = new System.Drawing.Point(10, 10)
            };

            errorPreviewPanel = new PictureBox
            {
                Size = new System.Drawing.Size(1340, 150),
                Location = new System.Drawing.Point(10, 40),
                BackColor = Color.FromArgb(50, 50, 50),
                BorderStyle = BorderStyle.FixedSingle,
                SizeMode = PictureBoxSizeMode.Zoom
            };

            previewContainer.Controls.Add(previewTitle);
            previewContainer.Controls.Add(errorPreviewPanel);
            this.Controls.Add(previewContainer);
        }

        private void CreateControlPanel()
        {
            controlPanel = new Panel
            {
                Size = new System.Drawing.Size(1370, 80),
                Location = new System.Drawing.Point(10, 640),
                BackColor = Color.FromArgb(45, 45, 45)
            };

            startC1Button = new Button
            {
                Text = "Inicio C1",
                Size = new System.Drawing.Size(200, 50),
                Location = new System.Drawing.Point(50, 15),
                BackColor = Color.FromArgb(80, 160, 80),
                ForeColor = Color.White,
                Font = new Font("Segoe UI", 14, FontStyle.Bold),
                FlatStyle = FlatStyle.Flat,
                Cursor = Cursors.Hand
            };
            startC1Button.FlatAppearance.BorderSize = 0;
            startC1Button.Click += StartC1Button_Click;

            startC2Button = new Button
            {
                Text = "Inicio C2",
                Size = new System.Drawing.Size(200, 50),
                Location = new System.Drawing.Point(270, 15),
                BackColor = Color.FromArgb(80, 160, 80),
                ForeColor = Color.White,
                Font = new Font("Segoe UI", 14, FontStyle.Bold),
                FlatStyle = FlatStyle.Flat,
                Cursor = Cursors.Hand
            };
            startC2Button.FlatAppearance.BorderSize = 0;
            startC2Button.Click += StartC2Button_Click;

            stopButton = new Button
            {
                Text = "Paro",
                Size = new System.Drawing.Size(150, 50),
                Location = new System.Drawing.Point(490, 15),
                BackColor = Color.FromArgb(200, 60, 60),
                ForeColor = Color.White,
                Font = new Font("Segoe UI", 14, FontStyle.Bold),
                FlatStyle = FlatStyle.Flat,
                Cursor = Cursors.Hand
            };
            stopButton.FlatAppearance.BorderSize = 0;
            stopButton.Click += StopButton_Click;

            analysisProgress = new ProgressBar
            {
                Size = new System.Drawing.Size(300, 20),
                Location = new System.Drawing.Point(700, 30),
                Style = ProgressBarStyle.Marquee,
                MarqueeAnimationSpeed = 30,
                Visible = false
            };

            controlPanel.Controls.Add(startC1Button);
            controlPanel.Controls.Add(startC2Button);
            controlPanel.Controls.Add(stopButton);
            controlPanel.Controls.Add(analysisProgress);
        }

        private void SetupRefreshTimer()
        {
            refreshTimer = new System.Windows.Forms.Timer
            {
                Interval = 10000 // 10 segundos
            };
            refreshTimer.Tick += RefreshTimer_Tick;
        }

        private async void StartC1Button_Click(object sender, EventArgs e)
        {
            statusLabel.Text = "● ANALIZANDO C1";
            statusLabel.ForeColor = Color.LimeGreen;
            analysisProgress.Visible = true;
            refreshTimer.Start();

            try
            {
                await cameraProcessor.StartAnalysis();
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error al iniciar análisis C1: {ex.Message}", "Error",
                               MessageBoxButtons.OK, MessageBoxIcon.Error);
                StopButton_Click(sender, e);
            }
        }

        private async void StartC2Button_Click(object sender, EventArgs e)
        {
            statusLabel.Text = "● ANALIZANDO C2";
            statusLabel.ForeColor = Color.LimeGreen;
            analysisProgress.Visible = true;
            refreshTimer.Start();

            try
            {
                await cameraProcessor.StartAnalysis();
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error al iniciar análisis C2: {ex.Message}", "Error",
                               MessageBoxButtons.OK, MessageBoxIcon.Error);
                StopButton_Click(sender, e);
            }
        }

        private void StopButton_Click(object sender, EventArgs e)
        {
            statusLabel.Text = "● DETENIDO";
            statusLabel.ForeColor = Color.Orange;
            analysisProgress.Visible = false;
            refreshTimer.Stop();

            cameraProcessor?.StopAnalysis();
        }

        private void RefreshTimer_Tick(object sender, EventArgs e)
        {
            UpdateDefectCounts();
            UpdateDefectLocationMap();
            UpdateErrorPreview();
        }

        public void AddDefect(string defectType, System.Drawing.Rectangle location, Bitmap defectImage)
        {
            if (defectCounts.ContainsKey(defectType))
            {
                defectCounts[defectType]++;

                var defect = new DetectedDefect
                {
                    Type = defectType,
                    Location = location,
                    Timestamp = DateTime.Now,
                    Image = defectImage
                };

                recentDefects.Add(defect);

                // Keep only recent defects (last 50)
                if (recentDefects.Count > 50)
                    recentDefects.RemoveAt(0);
            }
        }

        private void UpdateDefectCounts()
        {
            foreach (var card in defectCards)
            {
                if (defectCounts.ContainsKey(card.DefectType))
                {
                    card.UpdateCount(defectCounts[card.DefectType]);
                }
            }
        }

        private void UpdateDefectLocationMap()
        {
            if (recentDefects.Count == 0) return;

            var mapBitmap = new Bitmap(440, 260);
            using (var g = Graphics.FromImage(mapBitmap))
            {
                g.Clear(Color.FromArgb(70, 70, 70));

                // Draw coordinate grid
                using (var gridPen = new Pen(Color.FromArgb(100, 100, 100), 1))
                {
                    for (int x = 0; x < 440; x += 44)
                        g.DrawLine(gridPen, x, 0, x, 260);
                    for (int y = 0; y < 260; y += 26)
                        g.DrawLine(gridPen, 0, y, 440, y);
                }

                // Draw defects from last 10 seconds
                var recentDefectsInTime = recentDefects
                    .Where(d => DateTime.Now.Subtract(d.Timestamp).TotalSeconds <= 10)
                    .ToList();

                foreach (var defect in recentDefectsInTime)
                {
                    var color = GetDefectColor(defect.Type);
                    using (var brush = new SolidBrush(color))
                    {
                        // Scale location to map size
                        int x = (int)(defect.Location.X * 440.0 / 1920);
                        int y = (int)(defect.Location.Y * 260.0 / 1080);

                        g.FillEllipse(brush, x - 5, y - 5, 10, 10);
                        g.DrawString(defect.Type.Substring(0, 1),
                                   new Font("Arial", 8),
                                   Brushes.White, x - 3, y - 8);
                    }
                }
            }

            if (defectLocationPanel.InvokeRequired)
            {
                defectLocationPanel.Invoke(new Action(() =>
                {
                    defectLocationPanel.Image?.Dispose();
                    defectLocationPanel.Image = mapBitmap;
                }));
            }
            else
            {
                defectLocationPanel.Image?.Dispose();
                defectLocationPanel.Image = mapBitmap;
            }
        }

        private void UpdateErrorPreview()
        {
            var latestDefect = recentDefects.LastOrDefault();
            if (latestDefect?.Image != null)
            {
                if (errorPreviewPanel.InvokeRequired)
                {
                    errorPreviewPanel.Invoke(new Action(() =>
                    {
                        errorPreviewPanel.Image?.Dispose();
                        errorPreviewPanel.Image = latestDefect.Image;
                    }));
                }
                else
                {
                    errorPreviewPanel.Image?.Dispose();
                    errorPreviewPanel.Image = latestDefect.Image;
                }
            }
        }

        private Color GetDefectColor(string defectType)
        {
            return defectType switch
            {
                "Scratches" => Color.FromArgb(100, 150, 255),
                "Crack" => Color.FromArgb(255, 100, 100),
                "Spot" => Color.FromArgb(255, 200, 100),
                "Pore" => Color.FromArgb(150, 255, 150),
                "Seam" => Color.FromArgb(255, 150, 255),
                "Ocode" => Color.FromArgb(150, 255, 255),
                _ => Color.White
            };
        }

        protected override void OnFormClosed(FormClosedEventArgs e)
        {
            refreshTimer?.Stop();
            refreshTimer?.Dispose();

            // Desconectar eventos
            if (cameraProcessor != null)
            {
                cameraProcessor.DefectDetected -= OnDefectDetected;
                cameraProcessor.StopAnalysis();
            }

            foreach (var defect in recentDefects)
                defect.Image?.Dispose();

            base.OnFormClosed(e);
        }
    }

    // Helper classes
    public class DefectCountCard : Panel
    {
        private readonly Label typeLabel;
        private readonly Label countLabel;
        private readonly Label sizeLabel;
        public string DefectType { get; }

        public DefectCountCard(string defectType, Color cardColor)
        {
            DefectType = defectType;
            Size = new System.Drawing.Size(290, 140);
            BackColor = cardColor;
            BorderStyle = BorderStyle.FixedSingle;

            typeLabel = new Label
            {
                Text = defectType,
                Font = new Font("Segoe UI", 12, FontStyle.Bold),
                ForeColor = Color.White,
                Location = new System.Drawing.Point(10, 10),
                Size = new System.Drawing.Size(270, 25),
                TextAlign = ContentAlignment.TopCenter
            };

            countLabel = new Label
            {
                Text = "0",
                Font = new Font("Segoe UI", 36, FontStyle.Bold),
                ForeColor = Color.White,
                Location = new System.Drawing.Point(10, 40),
                Size = new System.Drawing.Size(270, 60),
                TextAlign = ContentAlignment.MiddleCenter
            };

            sizeLabel = new Label
            {
                Text = "0.0 mm",
                Font = new Font("Segoe UI", 10),
                ForeColor = Color.White,
                Location = new System.Drawing.Point(10, 105),
                Size = new System.Drawing.Size(270, 25),
                TextAlign = ContentAlignment.BottomCenter
            };

            Controls.Add(typeLabel);
            Controls.Add(countLabel);
            Controls.Add(sizeLabel);
        }

        public void UpdateCount(int count)
        {
            if (countLabel.InvokeRequired)
            {
                countLabel.Invoke(new Action(() => countLabel.Text = count.ToString()));
            }
            else
            {
                countLabel.Text = count.ToString();
            }
        }

        public void UpdateSize(double size)
        {
            if (sizeLabel.InvokeRequired)
            {
                sizeLabel.Invoke(new Action(() => sizeLabel.Text = $"{size:F1} mm"));
            }
            else
            {
                sizeLabel.Text = $"{size:F1} mm";
            }
        }
    }

    public class DetectedDefect
    {
        public string Type { get; set; }
        public System.Drawing.Rectangle Location { get; set; }
        public DateTime Timestamp { get; set; }
        public Bitmap Image { get; set; }
        public double Size { get; set; }
    }
}