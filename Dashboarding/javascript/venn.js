() => {
    const type = boolean('Euler', true);
    const data = object('Data', [
      { key: ['A'], data: 4 },
      { key: ['B'], data: 1 },
      { key: ['A', 'B'], data: 1 }
    ]);
  
    return (
      <VennDiagram
        type={type ? 'euler' : 'venn'}
        height={450}
        width={450}
        data={data}
      />
    );
  }