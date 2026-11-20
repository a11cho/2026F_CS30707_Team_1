"""저장된 추론 상태열을 관전자 GIF와 MJPEG AVI로 만듭니다. 모델 학습은 하지 않습니다."""
import argparse
from collections import Counter
from io import BytesIO
import json
from pathlib import Path
import struct
from PIL import Image,ImageDraw,ImageFont

FONT_PATH="C:/Windows/Fonts/malgun.ttf"
TILE_COLORS=((90,48,59),(38,67,100))
AGENT_COLORS=((255,115,135),(102,192,255))
ACTION_NAMES=("정지","이동 ↑","이동 ↓","이동 ←","이동 →","공격 ↑","공격 ↓","공격 ←","공격 →")
LABELS={"local_individual_gross_b0":"개인 보상 · local","local_team_terminal_b1":"팀 종료 보상 · local",
        "aware_individual_gross_b0":"개인 보상 · aware","aware_team_terminal_b1":"팀 종료 보상 · aware"}


def chunk(name,payload):
    return name+struct.pack("<I",len(payload))+payload+(b"\0" if len(payload)%2 else b"")


def write_avi(path,frames,fps):
    """외부 인코더 없이 표준 RIFF AVI 컨테이너에 JPEG 키프레임을 저장합니다."""
    jpeg=[]
    width,height=frames[0].size
    for frame in frames:
        out=BytesIO()
        frame.save(out,"JPEG",quality=88,subsampling=0)
        jpeg.append(out.getvalue())
    largest=max(map(len,jpeg))
    avih=struct.pack("<14I",int(1e6/fps),largest*fps,0,0x10,len(jpeg),0,1,largest,width,height,0,0,0,0)
    strh=struct.pack("<4s4sIHH8I4h",b"vids",b"MJPG",0,0,0,0,1,fps,0,len(jpeg),largest,0xffffffff,0,0,0,width,height)
    strf=struct.pack("<IiiHH4sIiiII",40,width,height,1,24,b"MJPG",width*height*3,0,0,0,0)
    header=chunk(b"LIST",b"hdrl"+chunk(b"avih",avih)+chunk(b"LIST",b"strl"+chunk(b"strh",strh)+chunk(b"strf",strf)))
    movie=[]
    index=[]
    offset=4
    for content in jpeg:
        part=chunk(b"00dc",content)
        index.append(struct.pack("<4sIII",b"00dc",0x10,offset,len(content)))
        movie.append(part)
        offset+=len(part)
    body=header+chunk(b"LIST",b"movi"+b"".join(movie))+chunk(b"idx1",b"".join(index))
    path.write_bytes(b"RIFF"+struct.pack("<I",len(body)+4)+b"AVI "+body)
    # 컨테이너 크기·인덱스 수와 JPEG 자체를 확인합니다.
    raw=path.read_bytes()
    assert struct.unpack_from("<I",raw,4)[0]==len(raw)-8
    assert len(index)==len(frames)
    for image in (jpeg[0],jpeg[-1]):
        with Image.open(BytesIO(image)) as decoded:
            assert decoded.size==(width,height)
            decoded.load()


def font(size):
    return ImageFont.truetype(FONT_PATH,size)


def panel(canvas,trace,index,origin,width=620,height=420):
    draw=ImageDraw.Draw(canvas)
    x,y=origin
    frame=trace["frames"][index]
    draw.rounded_rectangle((x,y,x+width,y+height),radius=12,fill="#ffffff",outline="#ccd5df",width=2)
    title=LABELS[trace["condition"]]+" / "+("MC" if trace["algorithm"]=="mc" else "SARSA(0)")
    draw.text((x+16,y+10),title,font=font(21),fill="#182b40")
    draw.text((x+16,y+38),f"학습 seed {trace['training_seed']} · 양 팀 모두 저장된 모델",font=font(15),fill="#596779")
    side=320
    rows,cols=len(frame["owner"]),len(frame["owner"][0])
    tile=side//cols
    bx,by=x+16,y+68
    for row in range(rows):
        for col in range(cols):
            draw.rectangle((bx+col*tile,by+row*tile,bx+(col+1)*tile,by+(row+1)*tile),fill=TILE_COLORS[frame["owner"][row][col]],outline="#263447",width=1)
    if index:
        previous={a["id"]:a for a in trace["frames"][index-1]["agents"]}
        for aid,action in frame["actions"].items():
            if action>=5 and previous[aid]["position"] is not None:
                row,col=previous[aid]["position"]
                dr,dc=((-1,0),(1,0),(0,-1),(0,1))[(action-1)%4]
                row,col=row+dr,col+dc
                if 0<=row<rows and 0<=col<cols:
                    draw.rectangle((bx+col*tile+4,by+row*tile+4,bx+(col+1)*tile-4,by+(row+1)*tile-4),outline="#f9d552",width=4)
    for agent in frame["agents"]:
        if agent["position"] is None:
            continue
        row,col=agent["position"]
        cx,cy=bx+col*tile+tile//2,by+row*tile+tile//2
        team=int(agent["id"].startswith("blue"))
        draw.ellipse((cx-24,cy-24,cx+24,cy+24),fill=AGENT_COLORS[team],outline="#ffffff",width=2)
        label=("B" if team else "R")+agent["id"][-1]
        draw.text((cx,cy),label,font=font(19),fill="#12202e",anchor="mm")
    tx=x+352
    draw.text((tx,y+72),f"틱 {frame['tick']:02d} / 64",font=font(21),fill="#172b40")
    draw.text((tx,y+107),f"빨강 {frame['areas'][0]}  :  파랑 {frame['areas'][1]}",font=font(21),fill="#172b40")
    counts=frame["counts"]
    draw.text((tx,y+142),f"점령 {counts.get('capture',0)} · 사망 {counts.get('death',0)}",font=font(16),fill="#596779")
    for i,agent in enumerate(frame["agents"]):
        aid=agent["id"]
        label=("B" if aid.startswith("blue") else "R")+aid[-1]
        state="대기" if agent["position"] is None else str(tuple(agent["position"]))
        action=ACTION_NAMES[frame["actions"][aid]] if aid in frame["actions"] else "—"
        draw.text((tx,y+181+i*29),f"{label}  {state}  {action}",font=font(16),fill="#172b40")
    draw.text((x+16,y+394),"금색 테두리: 직전 공격 목표  ·  R: 빨강 / B: 파랑",font=font(14),fill="#596779")


def render(root):
    traces=json.loads((root/"data/simulation_traces.json").read_text(encoding="utf-8"))
    output=root/"videos"
    output.mkdir(exist_ok=True)
    conditions=("local_individual_gross_b0","local_team_terminal_b1","aware_individual_gross_b0","aware_team_terminal_b1")
    sarsa=[next(t for t in traces if t["algorithm"]=="sarsa" and t["condition"]==c) for c in conditions]
    compare=[next(t for t in traces if t["algorithm"]==a and t["condition"]==conditions[0]) for a in ("mc","sarsa")]
    videos=[]
    for filename,chosen,rows,title in (("sarsa_four_conditions",sarsa,2,"네 조건 SARSA(0) · 최종 모델 양 팀 대결"),
                                       ("local_individual_mc_sarsa",compare,1,"개인 보상 · local · MC와 SARSA(0) 관전 비교")):
        frames=[]
        for index in range(65):
            canvas=Image.new("RGB",(1280,100+rows*436+48),"#eef3f8")
            draw=ImageDraw.Draw(canvas)
            draw.text((24,16),title,font=font(28),fill="#172b40")
            draw.text((24,57),"동일 환경 seed 7 · 정책 seed 1247478191 · 추론 전용 / 업데이트 없음",font=font(18),fill="#596779")
            for i,trace in enumerate(chosen):
                panel(canvas,trace,index,(16+(i%2)*632,100+(i//2)*436))
            draw.text((24,canvas.height-35),"고정된 한 장면의 정성 자료입니다. 전체 맵은 관전자용이며 모델은 local/aware 관측만 받습니다.",font=font(17),fill="#596779")
            frames.append(canvas)
        frames=[frames[0]]*2+frames+[frames[-1]]*8
        gif_frames=[im.resize((960,int(im.height*.75))).quantize(colors=192,method=Image.Quantize.FASTOCTREE,dither=Image.Dither.NONE) for im in frames]
        gif=output/(filename+".gif")
        gif_frames[0].save(gif,save_all=True,append_images=gif_frames[1:],loop=0,duration=250,disposal=2,optimize=False)
        write_avi(output/(filename+".avi"),frames,4)
        frames[34].save(output/(filename+"_poster.png"))
        with Image.open(gif) as animation:
            duration=0
            for index in range(animation.n_frames):
                animation.seek(index)
                duration+=animation.info.get("duration",0)
            assert duration==len(frames)*250,(duration,len(frames))
        videos.append(dict(name=filename,fps=4,duration_seconds=len(frames)/4,frame_count=len(frames),
                           avi_frame_count=len(frames),gif_stored_frames=animation.n_frames,gif_duration_ms=duration,
                           avi_resolution=list(frames[0].size),gif_resolution=list(gif_frames[0].size),
                           environment_seed=7,policy_seed=1247478191,training=False,
                           checkpoint_details=[{k:v for k,v in t.items() if k!="frames"} for t in chosen]))
        print("영상 저장 완료:",filename,flush=True)
    (output/"video_manifest.json").write_text(json.dumps(videos,ensure_ascii=False,indent=2),encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    args=parser.parse_args()
    if not args.root.resolve().is_relative_to(Path(__file__).resolve().parent):
        raise ValueError("영상은 smoke_test_mc_td 내부에만 저장할 수 있습니다.")
    render(args.root.resolve())
