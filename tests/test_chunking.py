from voxweave.chunking import pack_speech_segments


def test_packs_into_single_chunk_when_short():
    segs = [{"start": 0.0, "end": 2.0}, {"start": 3.0, "end": 5.0}]
    chunks = pack_speech_segments(segs, max_sec=240.0)
    assert chunks == [{"start": 0.0, "end": 5.0, "offset": 0.0}]


def test_splits_at_silence_when_exceeding_max():
    # three segments, each 100s, 1s gap; max=240 -> seg1+seg2 one chunk, seg3 one chunk
    segs = [
        {"start": 0.0, "end": 100.0},
        {"start": 101.0, "end": 201.0},
        {"start": 202.0, "end": 302.0},
    ]
    chunks = pack_speech_segments(segs, max_sec=240.0)
    assert len(chunks) == 2
    assert chunks[0] == {"start": 0.0, "end": 201.0, "offset": 0.0}
    assert chunks[1] == {"start": 202.0, "end": 302.0, "offset": 202.0}


def test_single_segment_longer_than_max_is_hard_split():
    # 500s continuous speech with no silence, max=240 -> hard cut (word cuts tolerated)
    segs = [{"start": 0.0, "end": 500.0}]
    chunks = pack_speech_segments(segs, max_sec=240.0)
    assert len(chunks) == 3
    assert chunks[0]["start"] == 0.0 and chunks[0]["end"] == 240.0
    assert chunks[1]["start"] == 240.0 and chunks[1]["end"] == 480.0
    assert chunks[2]["start"] == 480.0 and chunks[2]["end"] == 500.0
    assert [c["offset"] for c in chunks] == [0.0, 240.0, 480.0]


def test_empty_returns_empty():
    assert pack_speech_segments([], max_sec=240.0) == []
