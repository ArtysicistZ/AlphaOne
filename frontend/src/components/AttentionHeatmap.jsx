import React, { useEffect, useRef } from 'react';

// Viridis colormap — 256 sampled stops from matplotlib's viridis
// Each entry is [r, g, b] in 0–255 range
const VIRIDIS = [
  [68,1,84],[68,2,86],[69,4,87],[69,5,89],[70,7,90],[70,8,92],[70,10,93],[70,11,94],
  [71,13,96],[71,14,97],[71,16,99],[71,17,100],[71,19,101],[72,20,103],[72,22,104],
  [72,23,105],[72,24,106],[72,26,108],[72,27,109],[72,28,110],[72,29,111],[72,31,112],
  [72,32,113],[72,33,115],[72,35,116],[72,36,117],[72,37,118],[72,38,119],[72,40,120],
  [72,41,121],[71,42,122],[71,44,122],[71,45,123],[71,46,124],[71,47,125],[70,48,126],
  [70,50,126],[70,51,127],[69,52,128],[69,53,129],[69,55,129],[68,56,130],[68,57,131],
  [68,58,131],[67,60,132],[67,61,132],[66,62,133],[66,63,133],[66,64,134],[65,66,134],
  [65,67,135],[64,68,135],[64,69,136],[63,71,136],[63,72,137],[62,73,137],[62,74,137],
  [62,76,138],[61,77,138],[61,78,138],[60,79,139],[60,80,139],[59,82,139],[59,83,140],
  [58,84,140],[58,85,140],[57,86,141],[57,87,141],[56,89,141],[56,90,141],[55,91,142],
  [55,92,142],[54,93,142],[54,94,142],[53,95,142],[53,96,142],[52,97,143],[52,98,143],
  [51,99,143],[51,100,143],[50,101,143],[50,102,143],[49,103,143],[49,104,143],
  [49,105,143],[48,106,143],[48,107,143],[47,108,143],[47,109,143],[46,110,143],
  [46,111,143],[45,112,143],[45,113,143],[44,113,142],[44,114,142],[44,115,142],
  [43,116,142],[43,117,142],[42,118,142],[42,119,141],[42,120,141],[41,121,141],
  [41,122,141],[40,122,140],[40,123,140],[40,124,140],[39,125,139],[39,126,139],
  [39,127,139],[38,128,138],[38,129,138],[38,130,137],[37,131,137],[37,131,137],
  [37,132,136],[36,133,136],[36,134,135],[36,135,135],[35,136,134],[35,137,134],
  [35,137,133],[34,138,133],[34,139,132],[34,140,131],[33,141,131],[33,142,130],
  [33,143,129],[33,143,129],[32,144,128],[32,145,127],[32,146,127],[32,147,126],
  [32,148,125],[31,149,124],[31,149,123],[31,150,123],[31,151,122],[31,152,121],
  [31,153,120],[31,154,119],[31,154,118],[31,155,117],[31,156,116],[31,157,115],
  [31,158,114],[31,159,113],[31,159,112],[32,160,111],[32,161,110],[32,162,109],
  [33,163,108],[33,164,107],[34,164,106],[34,165,105],[35,166,104],[35,167,103],
  [36,168,102],[37,168,101],[37,169,99],[38,170,98],[39,171,97],[40,171,96],
  [41,172,95],[42,173,94],[43,174,93],[44,174,91],[46,175,90],[47,176,89],
  [48,176,88],[50,177,87],[51,178,85],[53,178,84],[54,179,83],[56,180,82],
  [57,180,80],[59,181,79],[61,181,78],[62,182,77],[64,183,75],[66,183,74],
  [68,184,73],[69,184,71],[71,185,70],[73,186,69],[75,186,67],[77,187,66],
  [79,187,65],[81,188,63],[83,188,62],[85,189,61],[87,189,59],[89,190,58],
  [91,190,57],[94,191,55],[96,191,54],[98,192,52],[100,192,51],[103,193,49],
  [105,193,48],[107,194,47],[109,194,45],[112,195,44],[114,195,42],[116,196,41],
  [119,196,39],[121,197,38],[123,197,37],[126,198,35],[128,198,34],[130,199,32],
  [133,199,31],[135,200,29],[138,200,28],[140,201,27],[142,201,25],[145,202,24],
  [147,202,22],[150,203,21],[152,203,20],[155,204,18],[157,204,17],[160,205,16],
  [162,205,15],[165,206,13],[167,206,12],[170,207,11],[172,207,10],[175,208,9],
  [177,208,8],[180,209,7],[182,209,6],[185,210,5],[187,210,5],[190,211,4],
  [192,211,4],[195,212,3],[197,212,3],[200,213,3],[202,213,3],[205,214,3],
  [207,214,3],[209,215,3],[212,215,4],[214,216,4],[217,216,5],[219,217,5],
  [221,217,6],[224,218,7],[226,218,8],[228,219,9],[231,219,11],[233,220,12],
  [235,220,14],[237,221,15],[239,221,17],[241,222,19],[243,222,21],[245,223,23],
  [247,224,25],[249,224,27],[250,225,30],[252,225,32],[253,226,35],[253,227,37],
];

function viridisColor(t) {
  const idx = Math.min(255, Math.max(0, Math.round(t * 255)));
  const [r, g, b] = VIRIDIS[idx];
  return `rgb(${r},${g},${b})`;
}

function AttentionHeatmap({ tokens, matrix }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!tokens || !matrix || tokens.length === 0) return;

    const canvas = canvasRef.current;
    const dpr = window.devicePixelRatio || 1;
    const ctx = canvas.getContext('2d');
    const n = tokens.length;

    const CELL_SIZE = 40;
    const LABEL_FONT_SIZE = 13;
    const LABEL_PADDING = 10;
    const COLORBAR_WIDTH = 16;
    const COLORBAR_GAP = 14;
    const COLORBAR_LABEL_WIDTH = 30;

    // Measure label width (use bold weight for worst-case width)
    ctx.font = `800 ${LABEL_FONT_SIZE}px "Space Grotesk", system-ui, sans-serif`;
    let maxLabelWidth = 0;
    for (const tok of tokens) {
      const w = ctx.measureText(tok).width;
      if (w > maxLabelWidth) maxLabelWidth = w;
    }

    const marginLeft = Math.ceil(maxLabelWidth) + LABEL_PADDING + 6;
    const marginTop = marginLeft;
    const gridSize = n * CELL_SIZE;
    const totalW = marginLeft + gridSize + COLORBAR_GAP + COLORBAR_WIDTH + COLORBAR_LABEL_WIDTH;
    const totalH = marginTop + gridSize + 12;

    // Set up HiDPI canvas
    canvas.width = totalW * dpr;
    canvas.height = totalH * dpr;
    canvas.style.width = totalW + 'px';
    canvas.style.height = totalH + 'px';
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, totalW, totalH);

    // Find global max for normalization (excluding diagonal self-attention)
    let globalMax = 0;
    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) {
        if (r !== c && matrix[r][c] > globalMax) globalMax = matrix[r][c];
      }
    }
    if (globalMax === 0) globalMax = 1;

    // Draw cells
    for (let row = 0; row < n; row++) {
      for (let col = 0; col < n; col++) {
        const raw = matrix[row][col];
        const t = Math.min(1, raw / globalMax);
        ctx.fillStyle = viridisColor(t);
        ctx.fillRect(
          marginLeft + col * CELL_SIZE,
          marginTop + row * CELL_SIZE,
          CELL_SIZE,
          CELL_SIZE,
        );
      }
    }

    // Grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.25)';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= n; i++) {
      ctx.beginPath();
      ctx.moveTo(marginLeft + i * CELL_SIZE, marginTop);
      ctx.lineTo(marginLeft + i * CELL_SIZE, marginTop + gridSize);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(marginLeft, marginTop + i * CELL_SIZE);
      ctx.lineTo(marginLeft + gridSize, marginTop + i * CELL_SIZE);
      ctx.stroke();
    }

    // Outer border
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.15)';
    ctx.lineWidth = 1;
    ctx.strokeRect(marginLeft, marginTop, gridSize, gridSize);

    // Axis labels
    const isTargetToken = (tok) => {
      const lower = tok.replace(/^[▁_]/, '').toLowerCase();
      return lower === 'target';
    };

    for (let i = 0; i < n; i++) {
      const tok = tokens[i];
      const highlight = isTargetToken(tok);
      const labelColor = highlight ? '#e03030' : '#1a2a3a';
      const labelFont = highlight
        ? `900 ${LABEL_FONT_SIZE}px "Space Grotesk", system-ui, sans-serif`
        : `600 ${LABEL_FONT_SIZE}px "Space Grotesk", system-ui, sans-serif`;

      const cellCenter = marginTop + i * CELL_SIZE + CELL_SIZE / 2;

      // Left axis
      ctx.font = labelFont;
      ctx.textBaseline = 'middle';
      ctx.textAlign = 'right';
      ctx.fillStyle = labelColor;
      ctx.fillText(tok, marginLeft - LABEL_PADDING, cellCenter);

      // Top axis (rotated) — front of word starts at grid top edge
      ctx.font = labelFont;
      const colCenter = marginLeft + i * CELL_SIZE + CELL_SIZE / 2;
      ctx.save();
      ctx.translate(colCenter, marginTop - 4);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = labelColor;
      ctx.fillText(tok, 0, 0);
      ctx.restore();
    }

    // Colorbar
    const barX = marginLeft + gridSize + COLORBAR_GAP;
    const barY = marginTop;
    const barH = gridSize;
    for (let y = 0; y < barH; y++) {
      const t = 1 - y / barH;
      ctx.fillStyle = viridisColor(t);
      ctx.fillRect(barX, barY + y, COLORBAR_WIDTH, 1);
    }

    ctx.strokeStyle = 'rgba(0, 0, 0, 0.2)';
    ctx.lineWidth = 1;
    ctx.strokeRect(barX, barY, COLORBAR_WIDTH, barH);

    // Colorbar labels
    ctx.fillStyle = '#1a2a3a';
    ctx.font = `500 10px "Space Grotesk", system-ui, sans-serif`;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText('High', barX + COLORBAR_WIDTH + 4, barY + 8);
    ctx.fillText('Low', barX + COLORBAR_WIDTH + 4, barY + barH - 8);
  }, [tokens, matrix]);

  if (!tokens || !matrix) return null;

  return (
    <div className="heatmap-scroll">
      <canvas
        ref={canvasRef}
        width={300}
        height={300}
        style={{ display: 'block' }}
      />
    </div>
  );
}

export default AttentionHeatmap;
