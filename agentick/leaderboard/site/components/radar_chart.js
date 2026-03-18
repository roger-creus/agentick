// Radar chart component using Plotly

function createRadarChart(elementId, agentData) {
    const capabilities = Object.keys(agentData.per_capability);
    const scores = capabilities.map(cap => agentData.per_capability[cap].mean_normalized_score);

    const data = [{
        type: 'scatterpolar',
        r: scores,
        theta: capabilities,
        fill: 'toself',
        name: agentData.agent_name
    }];

    const layout = {
        polar: {
            radialaxis: {
                visible: true,
                range: [0, 1]
            }
        },
        showlegend: true
    };

    Plotly.newPlot(elementId, data, layout);
}
