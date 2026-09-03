const express = require('express');
const cors = require('cors');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// Serve frontend static files
app.use(express.static(path.join(__dirname, '..', 'frontend')));

// API Routes
app.use('/api/auth', require('./routes/auth'));
app.use('/api/disasters', require('./routes/disasters'));
app.use('/api/teams', require('./routes/teams'));
app.use('/api/predict', require('./routes/prediction'));

// Serve frontend pages
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, '..', 'frontend', 'index.html'));
});

app.get('/admin', (req, res) => {
  res.sendFile(path.join(__dirname, '..', 'frontend', 'admin.html'));
});

app.get('/rescue', (req, res) => {
  res.sendFile(path.join(__dirname, '..', 'frontend', 'rescue.html'));
});

// Health check
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', uptime: process.uptime() });
});

app.listen(PORT, () => {
  console.log(`\n🚨 Disaster Management System Backend`);
  console.log(`   Server running on http://localhost:${PORT}`);
  console.log(`   Public Dashboard: http://localhost:${PORT}/`);
  console.log(`   Admin Panel:      http://localhost:${PORT}/admin`);
  console.log(`   Rescue Portal:    http://localhost:${PORT}/rescue`);
  console.log(`   API Health:       http://localhost:${PORT}/api/health\n`);
});
