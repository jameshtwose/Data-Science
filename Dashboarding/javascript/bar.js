import { BarChart } from 'reaviz';

const data = [
  { key: 'IDS', data: 14 },
  { key: 'Malware', data: 5 },
  { key: 'DLP', data: 18 }
];

const App = () => <BarChart width={350} height={250} data={data} />;
