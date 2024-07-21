const express = require('express');
const router = express.Router();
const { exec } = require('child_process');
const path = require('path');
const AWS = require('aws-sdk');
const multer = require('multer');
const multerS3 = require('multer-s3');

const s3 = new AWS.S3({
    accessKeyId: process.env.AWS_ACCESS_KEY_ID,
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY,
    region: process.env.AWS_REGION
});

const upload = multer({
    storage: multerS3({
        s3: s3,
        bucket: process.env.S3_BUCKET_NAME,
        acl: 'public-read',
        key: function (req, file, cb) {
            cb(null, Date.now().toString() + path.extname(file.originalname))
        }
    })
});

router.post('/predict', upload.single('image'), (req, res) => {
    const imageUrl = req.file.location;
    const command = `set PYTHONIOENCODING=utf-8 && python utils/preprocess.py "${imageUrl}"`;

    exec(command, { encoding: 'utf8' }, (error, stdout, stderr) => {
        if (error) {
            console.error(`Error executing script: ${error}`);
            return res.status(500).send('Internal Server Error');
        }
        console.log('Python script output:', stdout);
        try {
            const jsonString = stdout.trim();
            const result = JSON.parse(jsonString);
            if (result.error) {
                console.error(`Prediction error: ${result.error}`);
                res.status(500).send('Internal Server Error');
            } else {
                res.render('result', { result });
            }
        } catch (parseError) {
            console.error(`Error parsing JSON: ${parseError}`);
            res.status(500).send('Internal Server Error');
        }
    });
});

module.exports = router;
