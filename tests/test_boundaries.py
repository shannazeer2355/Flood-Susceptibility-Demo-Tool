import pytest

from flood_tool import boundaries


def test_all_kerala_districts_present():
    names = boundaries.district_names()
    assert len(names) == 14
    assert {"Thiruvananthapuram", "Ernakulam", "Wayanad", "Kasaragod"} <= set(names)


def test_district_lookup_ignores_case():
    aoi = boundaries.district(" ernakulam ")
    assert len(aoi) == 1
    area_km2 = aoi.to_crs(32643).area.iloc[0] / 1e6
    assert area_km2 == pytest.approx(3068, rel=0.05)


def test_unknown_district():
    with pytest.raises(ValueError, match="Unknown district"):
        boundaries.district("Atlantis")
