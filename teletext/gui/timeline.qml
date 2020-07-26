import QtQuick 2.12
import QtQuick.Controls 2.12
import QtGraphicalEffects 1.12

Rectangle {
    SystemPalette { id: myPalette; colorGroup: SystemPalette.Active }

    color: "black" //myPalette.window
    id: root

    TableView {
        property var selected: null
        property var hovered: null
        id: timeLine
        anchors.fill: parent
        leftMargin: 1
        topMargin: 1
        columnSpacing: 1
        rowSpacing: 1
        flickableDirection: Flickable.AutoFlickIfNeeded
        boundsBehavior: Flickable.StopAtBounds

        model: pyModel

        delegate: Rectangle {
            implicitWidth: 5
            implicitHeight: 5
            color: display

            MouseArea {
                anchors.fill: parent
                hoverEnabled: true
                onClicked: timeLine.selected = index
                //onEntered: parent.border.color = "transparent"
                //onExited: parent.border.color = "white"
            }
        }
    }
}