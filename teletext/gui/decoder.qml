import QtQuick 2.12
import QtGraphicalEffects 1.12


Rectangle {
    property int zoom: 2
    property int borderSize: 10 * zoom
    property bool crteffect: true
    property bool flashsrc: true
    width: teletext.width + borderSize * 4
    height: teletext.height + borderSize * 2
    border.width: borderSize
    border.color: "black"
    color: "black"
    Column {
        id: teletext
        objectName: "teletext"
        width: 40 * 8 * zoom
        height: 250 * zoom
        x: borderSize * 2
        y: borderSize
        Repeater {
            objectName: "rows"
            model: 25
            Row {
                Repeater {
                    objectName: "cols"
                    model: 40
                    Rectangle {
                        property string c: "X"
                        property int bg: 1
                        property int fg: 7
                        property bool dw: false
                        property bool dh: false
                        property bool glyph: false
                        property bool flash: false
                        color: ttpalette[bg]
                        Text {
                            renderType: Text.NativeRendering
                            x: (text[0]>="\uee20"&&text[0]<="\uee7f")?-zoom:0
                            color: ttpalette[fg]
                            text: c
                            font: ttfonts[(text[0]>="\uee20"&&text[0]<="\uee7f")?1:0][dw?1:0][dh?1:0]
                            MouseArea {
                                anchors.fill: parent
                                onClicked: teletext.currentIndex = index
                            }
                            visible: (!flash) || flashsrc
                        }
                        height: (dh?2:1) * 10 * zoom
                        width: (dw?2:1) * 8 * zoom
                        clip: true
                    }
                }
            }
        }
        layer.enabled: crteffect && (zoom > 1)
        layer.effect: ShaderEffect {
            fragmentShader: "
                    uniform lowp sampler2D source;
                    uniform lowp float qt_Opacity;
                    varying highp vec2 qt_TexCoord0;
                    varying lowp vec3 qt_FragCoord0;
                    void main() {
                        lowp vec4 tex = texture2D(source, qt_TexCoord0);
                        int zoom = " + zoom + ";
                        int row = int(gl_FragCoord.y) % zoom;
                        gl_FragColor = (0 < row && (row < 2 || row < (zoom-1))) ? tex : tex*0.6;
                    }
                "
        }
    }
    layer.enabled: crteffect && (zoom > 1)
    layer.effect: GaussianBlur {
        radius: 0.75 * zoom
    }
    SequentialAnimation on flashsrc {
        loops: -1
        running: true
        PropertyAction { value: false }
        PauseAnimation { duration: 333 }
        PropertyAction { value: true }
        PauseAnimation { duration: 1000 }
    }
}


