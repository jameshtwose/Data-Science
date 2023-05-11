// to run this file use the command: `node numbers_api.js` or `nodemon numbers_api.js`

const axios = require('axios');
const xml2js = require('xml2js');
const dotenv = require('dotenv');

dotenv.config();

const numberToWords = async (num = 21023) => {
  const url = 'https://number-conversion-service.p.rapidapi.com/webservicesserver/NumberConversion.wso';
  const headers = {
    'content-type': 'application/xml',
    'X-RapidAPI-Key': process.env.RAPID_API_KEY,
    'X-RapidAPI-Host': 'number-conversion-service.p.rapidapi.com'
  };
  const payload = `<?xml version='1.0' encoding='utf-8'?>
  <soap:Envelope xmlns:soap='http://schemas.xmlsoap.org/soap/envelope/'>
    <soap:Body>
      <NumberToWords xmlns='http://www.dataaccess.com/webservicesserver/'>
        <ubiNum>${num}</ubiNum>
      </NumberToWords>
    </soap:Body>
  </soap:Envelope>`;
  const response = await axios.post(url, payload, { headers });
  const data = await xml2js.parseStringPromise(response.data);
  const result = data['soap:Envelope']['soap:Body'][0]["m:NumberToWordsResponse"][0]["m:NumberToWordsResult"][0];
  console.log(result);
  return result;
};

numberToWords(43412);
