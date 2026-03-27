const puppeteer = require('puppeteer');
(async () => {
    const browser = await puppeteer.launch({headless: "new"});
    const page = await browser.newPage();
    await page.goto('file://' + __dirname + '/Task1_Report.html', {waitUntil: 'networkidle0'});
    await page.pdf({path: 'Task1_Report.pdf', format: 'A4', printBackground: true});
    await browser.close();
})();
