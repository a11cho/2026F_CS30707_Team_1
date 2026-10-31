import numpy as np

TEAM_COLORS = ((90, 48, 59), (38, 67, 100))
AGENT_COLORS = ((255, 115, 135), (102, 192, 255))


def render_rgb(snapshot, tile_size=24):
    """스냅샷을 (높이×tile_size, 너비×tile_size, 3) uint8 RGB 이미지로 반환한다.
    팀 소유 타일·격자·생존 에이전트만 그리며 창을 만들거나 게임 상태/RNG를 변경하지 않는다.
    """
    h, w = snapshot.owner.shape
    image = np.zeros((h*tile_size, w*tile_size, 3), dtype=np.uint8)
    for r in range(h):
        for c in range(w):
            image[r*tile_size:(r+1)*tile_size, c*tile_size:(c+1)*tile_size] = TEAM_COLORS[snapshot.owner[r, c]]
            image[r*tile_size, c*tile_size:(c+1)*tile_size] = (25, 30, 43)
            image[r*tile_size:(r+1)*tile_size, c*tile_size] = (25, 30, 43)
    inset = max(2, tile_size//4)
    for a in snapshot.agents:
        if a.alive:
            r, c = a.position
            image[r*tile_size+inset:(r+1)*tile_size-inset, c*tile_size+inset:(c+1)*tile_size-inset] = AGENT_COLORS[a.team]
    return image


class TkRenderer:
    """Snapshot-only renderer. Tk is imported only when a human window is requested."""
    def __init__(self, root=None, tile_size=36):
        """전달된 Tk root 또는 새 창에 타일맵 Canvas를 생성한다.
        GUI 없는 실행에서 Tk를 불러오지 않도록 생성자 내부에서만 tkinter를 import한다.
        """
        import tkinter as tk
        self.root = root or tk.Tk()
        self.root.title("Territory Control | Function Test")
        self.root.configure(bg="#121826")
        self.tile_size = tile_size
        self.canvas = tk.Canvas(self.root, bg="#121826", highlightthickness=0)
        self.canvas.pack(side="left", padx=14, pady=14)

    def draw(self, snapshot, selected=None):
        """스냅샷의 타일과 생존 에이전트를 Canvas에 다시 그린다.
        selected ID는 흰 테두리로 강조한다. 화면만 갱신하며 행동 제출이나 게임 전이는 수행하지 않는다.
        """
        size = self.tile_size
        h, w = snapshot.owner.shape
        self.canvas.configure(width=w*size, height=h*size)
        self.canvas.delete("all")
        for r in range(h):
            for c in range(w):
                color = "#%02x%02x%02x" % TEAM_COLORS[snapshot.owner[r, c]]
                self.canvas.create_rectangle(c*size, r*size, (c+1)*size, (r+1)*size,
                                             fill=color, outline="#1a2232")
        for a in snapshot.agents:
            if a.alive:
                r, c = a.position
                color = "#%02x%02x%02x" % AGENT_COLORS[a.team]
                self.canvas.create_oval(c*size+5, r*size+5, (c+1)*size-5, (r+1)*size-5,
                                        fill=color, outline="#ffffff" if a.agent_id == selected else "#243044",
                                        width=3 if a.agent_id == selected else 1)
                self.canvas.create_text((c+.5)*size, (r+.5)*size, text=str(int(a.agent_id[-1])+1),
                                        fill="#141925", font=("Consolas", 12, "bold"))

    def close(self):
        """사용 중인 Tk 창을 닫는다. 이미 파괴된 창의 TclError만 무시하며 다른 오류는 다시 전달한다."""
        try:
            self.root.destroy()
        except Exception as exc:
            import tkinter as tk
            if not isinstance(exc, tk.TclError):
                raise
